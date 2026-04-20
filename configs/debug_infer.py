# ── MX DEBUG SNAPSHOT ─────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from microxcaling.mx.specs import apply_mx_specs
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx.elemwise_ops import quantize_elemwise_op

# ── 1. Sanity checks ──────────────────────────────────────────────────────────
# self.in_ds is Sequential — the conv lives at index 0
conv = self.in_ds[0]
print("Sequential[0] type:", type(conv))      # expect MXConv2d
print("mx_none           :", conv.mx_none)     # expect False
print("mx_specs          :", conv.mx_specs)

specs = apply_mx_specs(conv.mx_specs)

# ── 2. Weight: original vs MX library ─────────────────────────────────────────
w = conv.weight.detach()
bf_w = quantize_elemwise_op(w, mx_specs=specs, round=specs['round_weight'])
w_mx = quantize_mx_op(bf_w, specs, elem_format=specs['w_elem_format'], axes=[1])

print("\n── Weight ──")
print("orig   | max:", w.abs().max().item(), "mean:", w.abs().mean().item())
print("mx lib | max:", w_mx.abs().max().item(), "mean:", w_mx.abs().mean().item())
print("err    | max:", (w - w_mx).abs().max().item(), "mean:", (w - w_mx).abs().mean().item())

# ── 3. Activation: original vs MX library ────────────────────────────────────
a = im_dwt_prev.detach()
bf_a = quantize_elemwise_op(a, mx_specs=specs, round=specs['round_output'])
a_mx = quantize_mx_op(bf_a, specs, elem_format=specs['a_elem_format'], axes=[1])

print("\n── Activation ──")
print("orig   | max:", a.abs().max().item(), "mean:", a.abs().mean().item())
print("mx lib | max:", a_mx.abs().max().item(), "mean:", a_mx.abs().mean().item())
print("err    | max:", (a - a_mx).abs().max().item(), "mean:", (a - a_mx).abs().mean().item())


# ── 4. Manual MX int8 quantization (max → shared scale → quant) ──────────────
# MX int8: shared_exp = floor(log2(max_abs)) per block-of-32 along axis,
#          step = 1/64,  clamp to [-127/64, 127/64]  (sign-magnitude, NOT ±128)
def manual_mx_int8(tensor, block_size=32, axis=1):
    t = tensor.detach().float()
    t = t.transpose(axis,
                    -1).contiguous()  # move quant axis to last dim
    orig_shape = t.shape
    n = t.shape[-1]
    pad = (block_size - n % block_size) % block_size
    if pad:
        t = F.pad(t, (0, pad))
    flat = t.reshape(-1,
                     block_size)  # (num_blocks, block_size)

    max_abs = flat.abs().amax(dim=-1, keepdim=True)
    safe = max_abs.clone();
    safe[max_abs == 0] = 1.0
    shared_exp = torch.floor(torch.log2(
        safe))  # E8M0 scale exponent
    scale = 2.0 ** shared_exp  # (num_blocks, 1)

    normed = flat / scale  # normalize to ~[-2, 2)
    q_int = torch.clamp(torch.round(normed * 64.0), -127.0, 127.0)  # ±127, sign-mag
    dq = (
                     q_int / 64.0) * scale  # dequantize

    if pad: dq = dq[..., :n]
    return dq.reshape(orig_shape).transpose(axis, -1).contiguous()


w_manual = manual_mx_int8(w, block_size=32, axis=1)
a_manual = manual_mx_int8(a, block_size=32, axis=1)

print("\n── Weight: manual vs MX library ──")
print("max diff:", (w_manual - w_mx).abs().max().item())
print("mean diff:", (w_manual - w_mx).abs().mean().item())

print("\n── Activation: manual vs MX library ──")
print("max diff:", (a_manual - a_mx).abs().max().item())
print("mean diff:", (a_manual - a_mx).abs().mean().item())

# ── 5. Output: FP32 vs MX ─────────────────────────────────────────────────────
out_fp32 = F.conv2d(a, w, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
out_mx   = self.in_ds(im_dwt_prev)
print("\n── Conv output: FP32 vs MX ──")
print("max diff:", (out_fp32 - out_mx).abs().max().item())
print("mean diff:", (out_fp32 - out_mx).abs().mean().item())
# ──────────────────────────────────────────────────────────────────────────────

# ── 6. Full model comparison: FP32 vs MX ──────────────────────────────────────
import torch

def compare_model_outputs(model_nr, model_nr_mx, *args, **kwargs):
    """Run both models on the same inputs and report output diffs."""
    with torch.no_grad():
        out_fp32 = model_nr(*args, **kwargs)
        out_mx   = model_nr_mx(*args, **kwargs)

    # Support tensor or tuple/list of tensors
    if isinstance(out_fp32, torch.Tensor):
        out_fp32 = (out_fp32,)
        out_mx   = (out_mx,)

    for i, (o_fp, o_mx) in enumerate(zip(out_fp32, out_mx)):
        diff = (o_fp - o_mx).abs()
        rel  = diff / (o_fp.abs() + 1e-8)
        print(f"output[{i}]  shape: {tuple(o_fp.shape)}")
        print(f"  fp32  | max: {o_fp.abs().max().item():.4f}  mean: {o_fp.abs().mean().item():.4f}")
        print(f"  mx    | max: {o_mx.abs().max().item():.4f}  mean: {o_mx.abs().mean().item():.4f}")
        print(f"  diff  | max: {diff.max().item():.4f}  mean: {diff.mean().item():.4f}  rel_mean: {rel.mean().item():.4f}")

# ── 7. Compare already-computed outputs ───────────────────────────────────────
for name, o_fp, o_mx in [
    ("pred_l1", pred_l1, pred_l1_mx),
    ("pred_l2", pred_l2, pred_l2_mx),
    ("pred_l3", pred_l3, pred_l3_mx),
    ("pred_g4", pred_g4, pred_g4_mx),
]:
    diff = (o_fp - o_mx).abs()
    rel  = diff / (o_fp.abs() + 1e-8)
    print(f"{name}  shape: {tuple(o_fp.shape)}")
    print(f"  fp32  | max: {o_fp.abs().max().item():.4f}  mean: {o_fp.abs().mean().item():.4f}")
    print(f"  mx    | max: {o_mx.abs().max().item():.4f}  mean: {o_mx.abs().mean().item():.4f}")
    print(f"  diff  | max: {diff.max().item():.4f}  mean: {diff.mean().item():.6f}  rel_mean: {rel.mean().item():.6f}")
    print()