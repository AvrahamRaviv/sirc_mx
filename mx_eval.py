"""Network-output SQNR between two models, on the same batches.

`measure_error` in the config reports per-layer output error, which answers a
different question: a layer can be badly damaged and still not move the network
output. `plan_mixed_precision` reports the network-level number, but only for
configs that run the ladder -- so a uniform config (every layer at one format)
has no way to produce a figure comparable with it.

This is that figure, built from the same primitives `_verify_assignment` uses,
so a number from here can be put straight beside `meta.assigned_vs_fp32`.
"""
import torch

import mx_sensitivity as _mxs


def net_sqnr(reference, candidate, data, forward_fn=None, batches=12,
             output_fn=None, log=None):
    """SQNR in dB of `candidate`'s network output against `reference`'s.

    Args:
        reference: the model whose output is treated as signal (usually FP32).
        candidate: the quantized model.
        data: an iterable of batches, e.g. a DataLoader. Only the first
            `batches` are used, and they are materialized once so a one-shot
            iterator is not consumed twice.
        forward_fn: forward_fn(model, batch), same one used for training.
        batches: how many batches to average over. 12 matches the planner's
            default so the two numbers are comparable.
        output_fn: how to reduce a model's return value to tensors. Defaults to
            the planner's flatten_outputs, which drops non-float tensors.

    Returns:
        {"sqnr_db", "mse", "cos_sim", "max_abs_err", "n_batches"}.
    """
    output_fn = output_fn or _mxs.flatten_outputs
    batch_list = _mxs.materialize(data, int(batches))
    if not batch_list:
        raise ValueError("net_sqnr got no batches")

    device = _mxs.model_device(candidate) or _mxs.model_device(reference)
    if device is not None:
        batch_list, _ = _mxs.to_device(batch_list, device)

    ref_training, cand_training = reference.training, candidate.training
    reference.eval()
    candidate.eval()
    acc = _mxs.ErrAcc()
    try:
        with torch.no_grad():
            for batch in batch_list:
                a = output_fn(_mxs.default_forward(reference, batch, forward_fn))
                b = output_fn(_mxs.default_forward(candidate, batch, forward_fn))
                if len(a) != len(b):
                    raise ValueError(
                        f"the two models returned different output counts "
                        f"({len(a)} vs {len(b)}); pass an output_fn that "
                        f"reduces them the same way")
                for ta, tb in zip(a, b):
                    acc.update(ta.detach().float().cpu(),
                               tb.detach().float().cpu())
    finally:
        if ref_training:
            reference.train()
        if cand_training:
            candidate.train()

    status, out = acc.summary()
    if status != _mxs.STATUS_OK or "sqnr_db" not in out:
        raise ValueError(
            f"could not measure the two models against each other: {status}. "
            f"An 'unreached' status means the forward never produced a float "
            f"output; check forward_fn and output_fn.")
    out = dict(out)
    out["status"] = status
    out["n_batches"] = len(batch_list)
    if log is not None:
        log.info(f"net_sqnr | {out['sqnr_db']:.2f} dB over "
                 f"{out['n_batches']} batches")
    return out
