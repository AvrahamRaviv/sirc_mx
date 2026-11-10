# DOF inference experiments: configs to make HW grids from

One config per experiment. Each makes its static HW grids once, offline; inference then reads them from
the config and never calibrates. Every change reuses the keys of an existing DOF config, named below.

| file | what it changes, from which config | grids written to `mx_config_grids.json` |
|---|---|---|
| `base.json` | the owner's DOF infer config, as is: the 30 convs on the HW path, output window `always` | `hw_grids` for the 30 convs |
| `se4.json` | `base.json` + the 4-bit shared-exponent register: `xblock_accum.in_exp_bits: 4`, `in_exp_mode: "always"` (from `mx_config_dof_run_win_se4.json`) | as base, each entry also holds `e_base_a` / `e_base_w` |
| `se4_warp.json` | `se4.json` + the `model.warp` `act_quant` entry (int8 ref, `calibrate: true`; Q8.8 grid; `call_sites: 2`) and `calibrate_act_scales` (both from `mx_config_dof_run_win_cal_convT.json`) | as se4, plus the warp ref scale per call site, written into the warp entry. The warp output is not quantized (no `output` fxp), though arch says it is plain INT8 on the NPE (`mx_config_npe_e4.json` quantizes it) |
| `se4_convT.json` | `se4.json` + the five ConvTranspose2d layers as in `mx_config_dof_run_win_cal_convT.json`: the dense `model.block{2,3,4}.conv1` on the HW path (`convT: "hw"`, already in base), `model.block{2,3}.convtranspose_2` on plain MX (group `convT_plain`) | as se4, plus `hw_grids` for the 3 `conv1` (plain MX needs none) |
| `se4_warp_convT.json` | `se4.json` + both changes above | both of the above |

Base and the 4-bit exponent:
- In `base.json` no ConvTranspose2d is in `layers`, so they all run FP32. Its `convT: "hw"` and its
  `convT_plain` group only take effect once those layers are listed, as in the two `convT` configs.
- 4-bit shared exponent = `in_exp_bits: 4`, the shift in [0, 15] above the calibrated `e_base` (the owner's
  meaning): each tensor's block exponents are held in 16 codes above it (`e_base_a` / `e_base_w`, written
  to the grids). The MX scale code width, `scale_bits`, stays 8. `mx_config_npe_e4.json`, a sim config,
  uses the other meaning, `scale_bits: 4`, which adds nothing to the grids.

Not made: 5 w_int8, 6 se4_w_int8: deferred by the owner 2026-11-10; weights stay MX int8 with the 4-bit
shared exponent, as in the se4 configs.

## Make the grids (GPU)

Per config: copy it into the run folder as `mx_config.json`, then run the trainer's quant step with data:

```python
model = MXQuantizer(save_dir).quant(model, data=cal_data, forward_fn=forward_fn)
```

`calibrate_hw_grids` (and, with the warp, `calibrate_act_scales`) runs on 8 batches and writes
`<save_dir>/mx_config_grids.json`; an existing file is kept, so start from an empty folder. Copy that file
over `mx_config.json` for inference.

`call_sites: 2` is measured: on the real DOF, `calibrate_act_scales` refused 3 with "called 2 times" (commit
`cb581b8`). If the model changes, it stops again and names the count: set `call_sites` to it.

Checked by `tests/test_infer_dof_configs.py`, on a CPU stand-in with DOF's layer names and kinds.
