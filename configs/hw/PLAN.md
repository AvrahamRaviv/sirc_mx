Plan: PT2E support for custom MX format with int24 static stage (POC)
0. Goal

PT2E flow for custom microscaling format, two-stage per edge:
> x → int24 quant/dequant (static, calibrated) → MX quant/dequant (dynamic) → float

MX kernel already exists (mx_quantizer / mx_dequantizer, below) — use it unchanged; this plan integrates around it. POC scope: simulation only, float version, no packed storage, no integer pipe modeling.

Existing kernel (copy verbatim into mxpt2e/mx_kernel.py):

> # <<< paste user's mx_quantizer / mx_dequantizer / MXTensor / mx_config_factory here, unchanged >>>
> # NOTE (POC, known limitations — do not fix now):
> # - floor() on the mantissa is biased for negative values; not round-to-nearest.
> #   Consequence: this kernel is NOT suitable for QAT (STE gradients will be off).
> #   Calibration-only (PTQ) flow is fine.
> # - config values are stored as torch tensors (magic-number style); works for POC.
> # - exponent window offset uses format constant e_min_layer=6; the per-tensor static
> #   info enters via the int24 stage only, not via an explicit anchor parameter.

1. Files (new package, do not modify torchao/pt2e internals)

> mxpt2e/
>   mx_kernel.py      # user's existing mx_quantizer/mx_dequantizer, verbatim + comment above
>   fake_quant.py     # Int24MXFakeQuant (int24 stage via standard FQ + MX noise)
>   quantizer.py      # MXQuantizer (annotate)
>   convert_pass.py   # post-convert_pt2e FX pass inserting mx_quantizer/mx_dequantizer
> tests/
>   test_prepare_convert.py
>   test_pass.py
>   test_parity.py

2. Phase 0 — API check (no code)

Grep the actual checkout, record findings in a note:

*    QuantizationSpec, QuantizationAnnotation, Quantizer, SharedQuantizationSpec import paths in installed torchao.
*    FakeQuantize, HistogramObserver in torch.ao.quantization.
*    In torchao/quantization/pt2e/convert.py: attributes the observer→quant-op replacement reads (dtype, qscheme, quant_min/quant_max), and whether dtype flows as qint32 or normalized int32.
*    convert_pt2e kwargs; exact emitted op names: quantized_decomposed.quantize_per_tensor[.tensor] / dequantize_per_tensor[.tensor].

Acceptance: note exists; nothing else.

3. Phase 1 — Observers + composite fake quant
Observers (two variants, same block-amax reduction)

Both subclasses below override only the stats-collection entry point: reshape input to blocks of block_size (pad last block with zeros, same as mx_kernel), take abs().amax(dim=-1), and feed per-block amaxes to the parent class's update logic. Everything else (qparams calc, buffers) is inherited.

> class BlockAmaxMinMaxObserver(MinMaxObserver):
>     """per-block amax stats, plain min/max (no averaging)"""
>     # forward: x_blocks = pad+reshape(x, block_size); super().forward(x_blocks.abs().amax(-1))
> 
> class BlockAmaxHistogramObserver(HistogramObserver):
>     """per-block amax stats, histogram search for clip"""
>     # same reduction, then super().forward(...)

* Shared config on both: quant_min=−(2^23), quant_max=2^23−1, qscheme=per_tensor_symmetric, dtype per Phase-0 finding.
* Selection registry (POC-simple):

> OBSERVER_REGISTRY = {
>     "minmax":    BlockAmaxMinMaxObserver,
>     "histogram": BlockAmaxHistogramObserver,
> }

* Expose the flat attributes convert reads (per Phase 0), delegating to the int24 stage.

Acceptance: for each observer type — fq(x) == manual two-stage application, bitwise; stats unaffected by MX stage. Plus: minmax scale == max_block_amax / (2^23−1) (sanity on known input).

Propagation to the rest of the plan:
* Phase 2 (quantizer): add observer_type: str = "minmax" (or "histogram") to MXQuantizer config; build the spec as

> obs_cls = OBSERVER_REGISTRY[observer_type]
> act_spec = QuantizationSpec(
>     ...,
>     observer_or_fake_quant_ctr=Int24MXFakeQuant.with_args(
>         observer=obs_cls.with_args(block_size=32),   # exact with_args nesting per Phase-0 API check
>         block_size=32,
>     ),
> )

(HistogramObserver may need extra ctor args like bins/upsample_rate in the with_args — check the signature in Phase 0 and pass defaults.)

*    Phase 5 acceptance: run test_prepare_convert.py and test_parity.py parametrized over both observer_type values — convert and the insertion pass must work identically regardless of which observer produced the scale.

Phase 2 — Quantizer: MXQuantizer + get_symmetric_quantization_mx_config

Target UX (must work exactly like this at the end of this phase):
> from mxpt2e.quantizer import MXQuantizer, get_symmetric_quantization_mx_config
> from torchao.quantization.pt2e import prepare_pt2e   # per Phase-0 path
> 
> quantizer = MXQuantizer()
> quantizer.set_global(
>     get_symmetric_quantization_mx_config(observer_type="minmax", block_size=32)
> )
> prepared = prepare_pt2e(model, quantizer)

get_symmetric_quantization_mx_config

Mirror the signatures inside the checkout's xnnpack_quantizer_utils.py (esp. QuantizationConfig field order — verify in Phase 0; do not trust positional fields). Conceptually:

> def get_symmetric_quantization_mx_config(
>     observer_type: str = "minmax",      # or "histogram" — keys of OBSERVER_REGISTRY (Phase 1)
>     block_size: int = 32,
> ) -> "QuantizationConfig":
>     obs_cls = OBSERVER_REGISTRY[observer_type]
> 
>     act_quantization_spec = QuantizationSpec(
>         dtype=...,                        # per Phase-0 finding (qint32 / int32)
>         quant_min=-(2**23),
>         quant_max=2**23 - 1,
>         qscheme=torch.per_tensor_symmetric,   # zp=0
>         is_dynamic=False,
>         observer_or_fake_quant_ctr=Int24MXFakeQuant.with_args(
>             observer=obs_cls.with_args(block_size=block_size),
>             block_size=block_size,
>         ),
>     )
> 
>     return QuantizationConfig(
>         input_activation=act_quantization_spec,
>         output_activation=act_quantization_spec,
>         weight=None,                      # POC: activations only → float weights/bias
>         bias=None,
>         is_qat=False,                     # MX kernel is not QAT-safe — hardcode False
>     )

Notes for implementer:

*    is_qat=False (non-negotiable). Calibration still works: prepare attaches Int24MXFakeQuant, whose internal observer collects stats in train mode while forward applies int24+MX noise — that is our whole simulation loop.
*    No validation of dtype inside this factory: the (dtype, qscheme) combo must survive prepare's normalization checks — that's a Phase-0 item, not validation here.

MXQuantizer

Subclass XNNPACKQuantizer; do not fork its annotator functions — they never instantiate observers (prepare does, via the spec), so the int24+MX spec flows through them untouched. Only override what's backend-specific for mx_xnnpack:

> class MXQuantizer(XNNPACKQuantizer):
>     """PT2E quantizer for backend 'mx_xnnpack': int24 static stage + MX dynamic stage."""
> 
>     # NOTE: adjust to however this checkout keys per-backend op tables
>     #       (grep XNNPACKQuantizer's op list / backend attribute in Phase 0).
>     # POC op coverage: start with the same ops mx_xnnpack supports — minimum: linear, add(?).
>     SUPPORTED_OPS = [...]   # per Phase-0 finding; do not hardcode blindly
> 
>     # inherits set_global / annotate / pattern registration

If the checkout has no backend-keyed op table (older torchao versions don't), then MXQuantizer = XNNPACKQuantizer subclass with zero overrides other than the name/docstring — write a comment saying so, and let opencode confirm rather than invent structure.

Acceptance:

1.    Usage snippet above runs on the 2-layer Linear-ReLU toy model; prepared graph contains Int24MXFakeQuant modules at annotated edges (matching exactly the pattern set SUPPORTED_OPS covers — no surprises).
2.    Calibration step: calling prepared model on data updates each Int24MXFakeQuant.activation_post_process; resulting scale in (0, ∞) and zp == 0.
3.    Both observer_type values parametrize the whole test.
4.    A normal XNNPACKQuantizer on the same model is unaffected (no import-time side effects, separate registries).

Additions to Phase 0 (append these greps)

*    xnnpack_quantizer.py: how does XNNPACKQuantizer restrict which ops get annotated (op table / _XNNPACK_OPS / per-op annotator dict)? Is there any "xnnpack" backend string keying anywhere (i.e., does mx_xnnpack need to be registered, or is it just a name)?
*    QuantizationConfig exact fields/order and whether weight=None is tolerated by all annotators that would run (if an annotator dereferences config.weight, either give it a float spec or drop that op from SUPPORTED_OPS — record the choice).
*    set_global implementation: confirm it just stores config (no dtype validation).

Downstream propagation

    Phase 3 acceptance (convert) and Phase 4 (pass): unchanged — both operate purely off observer attributes / emitted ops, neither knows the quantizer name. Add one line to Phase-3 acceptance: "convert+pass results identical whether the model was prepared via MXQuantizer(minmax) or MXQuantizer(histogram)."
    README: usage snippet above is the API surface; document observer_type and block_size knobs and the PTQ-only restriction (no is_qat=True).

Phase 3 — Standard convert (stage 1 only)

    Calibrate toy model; convert_pt2e(model, use_reference_representation=False) (reference representation forbidden — assumes 8-bit patterns).
    Check graph: quantize_per_tensor*/dequantize_per_tensor* nodes present with quant_min=−8388608, quant_max=8388607; dtype accepted by the op (if qint32 leaks to the op — the one expected break point — smallest local shim, document it).

Acceptance (test_prepare_convert.py): converted model runs; stage-1 q/dq numerics match prepared model exactly (hook-based comparison).
6. Phase 4 — Post-convert pass

    Walk the converted graph; for every int24 dequantize_per_tensor[.default|.tensor] node (match both overloads), insert the existing functions as a pair: mx_dequantizer(mx_quantizer(dq_out, block_size=32)) immediately after the dq (after any trailing dtype-cast node if present).
    POC simplification: insert as call_function nodes on the Python functions (graph won't be exportable — fine for POC; note it).
    MX params (block_size, mx_type) are constants on the inserted nodes; no need to derive anything from the dq node — the int24 range already constrains the input. graph.lint(); recompile() after edits.

Acceptance (test_pass.py): toy model gets one mx_quantizer→mx_dequantizer pair per int24 dq; non-int24 dqs untouched (interleave an int8-quantized op in toy model to prove selectivity).
7. Phase 5 — Parity + regression

    test_parity.py: prepared-calibrated vs convert+pass model — bitwise equal outputs on 100 random batches (both implement identical int24→MX numerics by construction; divergence = wrong insertion point).
    Regression: feed calibration-range inputs, confirm shared_exp stays in range and data within int8 limits in practice.

8. Do-NOT list

    No new convert function; stock convert_pt2e(..., use_reference_representation=False) only.
    No stats collection after MX noise (raw input only).
    No QAT / no training through these graphs (kernel not QAT-safe — see comments).
    No changes inside torchao/pt2e.
    No asymmetric (zp≠0) int24 configs.

9. Final gate

pytest tests/ -q all green; parity test bitwise.
