"""npe_mx_emu: standalone NPE hardware emulation of the MX format.

Pipeline: MX quantizers (mx_format) -> TRU/BCU/PCU (hw_pipe) -> calibrated
conv wrapper (layers).  Calibration via observers (observers) + calib glue.
"""

from npe_mx_emu.mx_format import (
    GREATER_THAN_ONE_BIT,
    SIGN_BIT,
    MXTensor,
    MXTensorConfig,
    base_mx_quantizer,
    input_mx_quantizer,
    mantissa_offset,
    mx_config_factory,
    mx_dequantizer,
    mx_weight_quantizer,
)
from npe_mx_emu.hw_pipe import (
    Cstruct,
    MXConv2D_Loops,
    align_mx_to_elements,
    bcu,
    pcu,
    pcu_div,
    pcu_pla,
    relux,
    tru_a,
    tru_w,
)
from npe_mx_emu.calib import (
    LayerMXQParams,
    build_c_struct,
    build_c_struct_arr,
    e_min_layer,
)
from npe_mx_emu.observers import MXObserver
from npe_mx_emu.layers import CalibratedMXConv2d

__all__ = [
    "Cstruct",
    "CalibratedMXConv2d",
    "GREATER_THAN_ONE_BIT",
    "LayerMXQParams",
    "MXConv2D_Loops",
    "MXObserver",
    "MXTensor",
    "MXTensorConfig",
    "SIGN_BIT",
    "align_mx_to_elements",
    "base_mx_quantizer",
    "bcu",
    "build_c_struct",
    "build_c_struct_arr",
    "e_min_layer",
    "input_mx_quantizer",
    "mantissa_offset",
    "mx_config_factory",
    "mx_dequantizer",
    "mx_weight_quantizer",
    "pcu",
    "pcu_div",
    "pcu_pla",
    "relux",
    "tru_a",
    "tru_w",
]
