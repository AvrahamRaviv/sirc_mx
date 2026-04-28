import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, '/Users/avrahamraviv/PycharmProjects')
sys.path.insert(0, '/home/avrahamra/PycharmProjects')

from microxcaling.mx import MxSpecs
from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.linear import Linear as MXLinear

from mx_layers_blocked import MXConv2dBlocked, MXLinearBlocked


def _specs(block_size=32, custom_cuda=False):
    sp = MxSpecs()
    sp['w_elem_format'] = 'int8'
    sp['a_elem_format'] = 'int8'
    sp['block_size'] = block_size
    sp['scale_bits'] = 8
    sp['shared_exp_method'] = 'max'
    sp['custom_cuda'] = custom_cuda
    return sp


def _attach_xblock_attrs(layer, enabled=True, bits=48, backend='python',
                          scale_exp=None, saturate=True, ste_mask=False):
    layer.xblock_accum = {
        'enabled': enabled,
        'bits': bits,
        'backend': backend,
        'scale_exp': scale_exp,
        'saturate': saturate,
        'ste_mask': ste_mask,
    }


def _build_pair(in_feat, out_feat, bias, seed=0):
    torch.manual_seed(seed)
    sp = _specs()
    ref = MXLinear(in_feat, out_feat, bias=bias, mx_specs=sp)
    blk = MXLinearBlocked(in_feat, out_feat, bias=bias, mx_specs=sp)
    with torch.no_grad():
        blk.weight.copy_(ref.weight)
        if bias:
            blk.bias.copy_(ref.bias)
    _attach_xblock_attrs(blk)
    return ref, blk


@pytest.mark.parametrize("in_feat,out_feat", [(64, 32), (128, 64), (1024, 256)])
@pytest.mark.parametrize("bias", [True, False])
def test_parity_vs_mxlinear_at_48bit(in_feat, out_feat, bias):
    """Blocked at fixed_point/48b must match flat MXLinear to FP32 precision.

    Same quantization applied, same FLOPs, only reduction order differs.
    """
    ref, blk = _build_pair(in_feat, out_feat, bias, seed=7)
    x = torch.randn(8, in_feat)
    ref.eval(); blk.eval()
    with torch.no_grad():
        y_ref = ref(x)
        y_blk = blk(x)
    # Only reduction order differs (F.linear one-shot vs blocked einsum + sum).
    # FP32 accumulation is non-associative, so tolerance scales with in_feat.
    scale = max(y_ref.abs().max().item(), 1.0)
    assert torch.allclose(y_ref, y_blk, atol=1e-3 * scale, rtol=1e-3), \
        f"max diff={(y_ref - y_blk).abs().max().item():.3e}  scale={scale:.3e}"


def test_parity_3d_input():
    """Linear accepts (B, seq, in) — blocked path must preserve that."""
    ref, blk = _build_pair(64, 32, bias=True, seed=2)
    x = torch.randn(4, 10, 64)
    ref.eval(); blk.eval()
    with torch.no_grad():
        y_ref = ref(x)
        y_blk = blk(x)
    assert y_blk.shape == (4, 10, 32)
    assert torch.allclose(y_ref, y_blk, atol=1e-4, rtol=1e-4)


def test_assert_on_indivisible_in_features():
    sp = _specs(block_size=32)
    blk = MXLinearBlocked(48, 16, bias=False, mx_specs=sp)
    _attach_xblock_attrs(blk)
    with pytest.raises(AssertionError):
        blk(torch.randn(2, 48))


def test_grad_flows_through_blocked_path():
    ref, blk = _build_pair(64, 32, bias=True, seed=3)
    x = torch.randn(4, 64, requires_grad=True)
    y = blk(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_dispatch_default_fp32_uses_original_mxlinear(tmp_path):
    """With default xblock_accum_mode='fp32', _replace_layers instantiates original MXLinear."""
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
        },
        "layers": [{"name": "fc"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32)
        def forward(self, x):
            return self.fc(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    assert isinstance(model.fc, MXLinear)
    assert not isinstance(model.fc, MXLinearBlocked)


def test_dispatch_fixed_point_uses_blocked(tmp_path):
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
            "xblock_accum": {
                "enabled": True,
                "bits": 48,
                "backend": "python",
            },
        },
        "layers": [{"name": "fc"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(64, 32)
        def forward(self, x):
            return self.fc(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    assert isinstance(model.fc, MXLinearBlocked)
    # forward runs without error
    model.fc(torch.randn(2, 64))


# ======================================================================
# Conv2d
# ======================================================================

def _build_conv_pair(in_ch, out_ch, kernel, stride=1, padding=0, bias=True, seed=0):
    torch.manual_seed(seed)
    sp = _specs()
    ref = MXConv2d(in_ch, out_ch, kernel, stride=stride, padding=padding,
                   bias=bias, mx_specs=sp)
    blk = MXConv2dBlocked(in_ch, out_ch, kernel, stride=stride, padding=padding,
                          bias=bias, mx_specs=sp)
    with torch.no_grad():
        blk.weight.copy_(ref.weight)
        if bias:
            blk.bias.copy_(ref.bias)
    _attach_xblock_attrs(blk)
    return ref, blk


@pytest.mark.parametrize("in_ch,out_ch,kernel,stride,padding", [
    (32, 16, 1, 1, 0),
    (32, 16, 3, 1, 1),
    (64, 32, 3, 2, 1),
    (64, 32, 7, 2, 3),
    (256, 128, 3, 1, 1),
])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_parity_vs_mxconv2d_at_48bit(in_ch, out_ch, kernel, stride, padding, bias):
    ref, blk = _build_conv_pair(in_ch, out_ch, kernel, stride, padding, bias, seed=11)
    x = torch.randn(2, in_ch, 16, 16)
    ref.eval(); blk.eval()
    with torch.no_grad():
        y_ref = ref(x)
        y_blk = blk(x)
    assert y_blk.shape == y_ref.shape
    scale = max(y_ref.abs().max().item(), 1.0)
    assert torch.allclose(y_ref, y_blk, atol=1e-3 * scale, rtol=1e-3), \
        f"max diff={(y_ref - y_blk).abs().max().item():.3e}  scale={scale:.3e}"


def test_conv_assert_on_indivisible_channels():
    sp = _specs(block_size=32)
    blk = MXConv2dBlocked(48, 16, 3, padding=1, bias=False, mx_specs=sp)
    _attach_xblock_attrs(blk)
    with pytest.raises(AssertionError):
        blk(torch.randn(2, 48, 8, 8))


def test_conv_grad_flow():
    ref, blk = _build_conv_pair(32, 16, 3, padding=1, bias=True, seed=5)
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    y = blk(x)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_conv_dispatch_fixed_point_uses_blocked(tmp_path):
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
            "xblock_accum": {
                "enabled": True,
                "bits": 48,
                "backend": "python",
            },
        },
        "layers": [{"name": "conv"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 16, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    assert isinstance(model.conv, MXConv2dBlocked)
    model.conv(torch.randn(1, 32, 8, 8))


def test_conv_dispatch_default_fp32_uses_original(tmp_path):
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
        },
        "layers": [{"name": "conv"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 16, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    assert isinstance(model.conv, MXConv2d)
    assert not isinstance(model.conv, MXConv2dBlocked)


def test_conv_nondivisible_channels_falls_back_to_original(tmp_path, capsys):
    """C not divisible by bs → dispatcher falls back to original MXConv2d + prints name."""
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
            "xblock_accum": {
                "enabled": True,
                "bits": 48,
                "backend": "python",
                "verbose": 2,
            },
        },
        "layers": [{"name": "conv"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
        def forward(self, x):
            return self.conv(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    out = capsys.readouterr().out
    assert isinstance(model.conv, MXConv2d)
    assert not isinstance(model.conv, MXConv2dBlocked)
    assert "blocked path skipped for conv 'conv'" in out
    assert "in_channels=3" in out


def test_linear_nondivisible_features_falls_back_to_original(tmp_path, capsys):
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
            "xblock_accum": {
                "enabled": True,
                "bits": 48,
                "backend": "python",
                "verbose": 2,
            },
        },
        "layers": [{"name": "fc"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(100, 32)
        def forward(self, x):
            return self.fc(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    out = capsys.readouterr().out
    assert isinstance(model.fc, MXLinear)
    assert not isinstance(model.fc, MXLinearBlocked)
    assert "blocked path skipped for linear 'fc'" in out
    assert "in_features=100" in out


def test_conv_groups_falls_back_to_original(tmp_path):
    """groups != 1 unsupported by blocked path; dispatcher keeps original MXConv2d."""
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
            "xblock_accum": {
                "enabled": True,
                "bits": 48,
                "backend": "python",
            },
        },
        "layers": [{"name": "conv"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        def forward(self, x):
            return self.conv(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    assert isinstance(model.conv, MXConv2d)
    assert not isinstance(model.conv, MXConv2dBlocked)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
