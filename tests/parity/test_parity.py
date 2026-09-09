"""Reference-parity tests: OLM architectures vs their ``transformers`` twins.

Run with::

    pytest tests/parity -q

Everything is CPU-only, offline, and randomly initialised at a tiny scale.
"""

from __future__ import annotations

import pytest
import torch

from ._harness import (
    causal_lm_loss,
    check_coverage,
    load_reference_weights,
    randomize_degenerate_parameters,
    run_parity,
)
from ._spec import ParityCase
from .architectures import ALL_CASES, CASES_BY_NAME
from .architectures import llama as llama_module

#: Cases whose reference is deliberately left at a default that OLM does not
#: implement. They are asserted to *disagree*, which doubles as a check that the
#: harness is capable of detecting a real discrepancy.
KNOWN_DIVERGENT = {"gpt2-stock", "gemma2-stock"}

MATCHING_CASES = [c for c in ALL_CASES if c.name not in KNOWN_DIVERGENT]
DIVERGENT_CASES = [c for c in ALL_CASES if c.name in KNOWN_DIVERGENT]

DTYPES = [torch.float64, torch.float32]

#: Tolerances are set well above the largest value the suite actually measures
#: (~3e-7), and are deliberately *not* tuned per architecture. OLM's LayerNorm
#: and RMSNorm both upcast to float32 internally and cast back, so float64 runs
#: cannot resolve below roughly float32 epsilon either; both dtypes therefore
#: share one tolerance.
LOGIT_TOL = 1e-5
LOSS_TOL = 1e-6
GRAD_COSINE_TOL = 1e-9


def _ids(case: ParityCase) -> str:
    return case.name


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids)
def test_weight_map_is_complete(case: ParityCase) -> None:
    """Every parameter on both sides must be named in the mapping.

    This is what stops a parity number from being accidentally meaningless: an
    unmapped OLM parameter would silently keep its random initialisation.
    """
    torch.manual_seed(0)
    hf = case.build_hf()
    olm = case.build_olm()
    check_coverage(case, olm, hf, list(case.build_map(olm, hf)))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d).replace("torch.", ""))
@pytest.mark.parametrize("case", MATCHING_CASES, ids=_ids)
def test_matches_reference(case: ParityCase, dtype: torch.dtype) -> None:
    result = run_parity(case, dtype=dtype)

    assert result.max_abs_logit_diff < LOGIT_TOL, (
        f"{case.name}/{result.dtype}: max |logit difference| "
        f"{result.max_abs_logit_diff:.3e} exceeds {LOGIT_TOL:.0e} "
        f"(reference logit scale {result.logit_scale:.3f})"
    )
    assert result.max_abs_loss_diff < LOSS_TOL, (
        f"{case.name}/{result.dtype}: max |loss difference| "
        f"{result.max_abs_loss_diff:.3e} exceeds {LOSS_TOL:.0e}"
    )
    assert result.grad_cosine_sim > 1.0 - GRAD_COSINE_TOL, (
        f"{case.name}/{result.dtype}: gradient cosine similarity "
        f"{result.grad_cosine_sim:.12f} below 1 - {GRAD_COSINE_TOL:.0e}"
    )
    assert result.min_per_tensor_grad_cosine > 1.0 - 1e-6, (
        f"{case.name}/{result.dtype}: worst per-parameter gradient cosine "
        f"{result.min_per_tensor_grad_cosine:.9f} on {result.worst_grad_tensor}"
    )


#: Each divergent case and the matched-activation case it is compared against.
DIVERGENCE_PAIRS = {"gpt2-stock": "gpt2", "gemma2-stock": "gemma2"}

#: How much larger the stock-reference disagreement must be than the
#: matched-reference one before we call it a real difference rather than noise.
DIVERGENCE_RATIO = 10.0


@pytest.mark.parametrize("case", DIVERGENT_CASES, ids=_ids)
def test_known_divergence_is_detected(case: ParityCase) -> None:
    """OLM does not reproduce these references, and the harness must say so.

    OLM's ClassicFFN and GeGLU both use exact erf GELU, while stock GPT-2 uses
    ``gelu_new`` and stock Gemma 2 uses ``gelu_pytorch_tanh``. The divergence is
    small in absolute terms, so it is judged against the *same architecture with
    the activation aligned* rather than against a fixed tolerance: that isolates
    the activation as the cause and keeps the finding from quietly regressing
    into a pass.
    """
    baseline_name = DIVERGENCE_PAIRS[case.name]
    baseline = run_parity(CASES_BY_NAME[baseline_name], dtype=torch.float64)
    result = run_parity(case, dtype=torch.float64)

    ratio = result.max_abs_logit_diff / max(baseline.max_abs_logit_diff, 1e-300)
    assert ratio > DIVERGENCE_RATIO, (
        f"{case.name} matched its stock reference about as well as "
        f"{baseline_name} matched its activation-aligned one "
        f"({result.max_abs_logit_diff:.3e} vs {baseline.max_abs_logit_diff:.3e}); "
        "the documented activation-function difference may have been fixed, in "
        "which case move this case out of KNOWN_DIVERGENT."
    )


def test_harness_detects_a_broken_mapping() -> None:
    """Negative control: corrupting the mapping must break parity.

    Without this, a harness that accidentally compared a model to itself would
    report perfect agreement forever. Here the RoPE head permutation is dropped
    from Llama 3's Q/K projections -- the single subtlest part of the whole
    mapping -- and parity must collapse.
    """
    import dataclasses

    from ._spec import MapEntry

    case = llama_module.CASES[2]
    assert case.name == "llama3"

    def broken_map(olm, hf) -> list[MapEntry]:
        return [
            dataclasses.replace(e, fn=None, grad_fn=None)
            if (".q_proj.weight" in e.olm or ".k_proj.weight" in e.olm)
            else e
            for e in case.build_map(olm, hf)
        ]

    broken = dataclasses.replace(case, name="llama3-broken", build_map=broken_map)
    result = run_parity(broken, dtype=torch.float64)

    assert result.max_abs_logit_diff > 1e-3, (
        "dropping the RoPE convention permutation did not change the logits; "
        "the harness is not measuring what it claims to measure"
    )


@pytest.mark.parametrize("case", ALL_CASES, ids=_ids)
def test_reference_constants_are_randomized(case: ParityCase) -> None:
    """Norm scales and biases must not still be constant when we compare.

    ``transformers`` initialises them to exactly ones or zeros, and constant
    vectors are permutation-invariant, so leaving them alone would hide any
    mapping bug affecting them -- which is exactly how Qwen3's QK-norm
    permutation went unnoticed until the gradient check caught it.
    """
    torch.manual_seed(0)
    hf = case.build_hf()
    randomize_degenerate_parameters(hf, seed=2)

    still_constant = [
        name
        for name, p in hf.named_parameters(remove_duplicate=True)
        if p.numel() > 1 and bool(torch.all(p.reshape(-1) == p.reshape(-1)[0]))
    ]
    assert not still_constant, (
        f"{case.name}: parameters left at a constant value, so a permutation "
        f"error in them would be undetectable: {still_constant}"
    )


def test_loss_is_computed_identically_for_both_models() -> None:
    """The loss difference must reflect the logit difference and nothing else."""
    logits = torch.randn(2, 8, 11, dtype=torch.float64)
    labels = torch.randint(0, 11, (2, 8))
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 11), labels[:, 1:].reshape(-1)
    )
    assert torch.equal(causal_lm_loss(logits, labels), expected)


def test_reference_weights_actually_land_in_the_olm_model() -> None:
    """Guard the copy itself: a no-op ``load`` would fake perfect parity."""
    case = MATCHING_CASES[0]
    torch.manual_seed(0)
    hf = case.build_hf().to(torch.float64)
    olm = case.build_olm().to(torch.float64)
    entries = list(case.build_map(olm, hf))

    olm_params = dict(olm.named_parameters(remove_duplicate=True))
    entry = entries[0]
    before = olm_params[entry.olm].detach().clone()

    load_reference_weights(olm, hf, entries)
    after = olm_params[entry.olm].detach()

    hf_params = dict(hf.named_parameters(remove_duplicate=True))
    expected = entry.apply(*[hf_params[n].detach() for n in entry.hf_names])
    assert torch.equal(after, expected)
    assert not torch.equal(after, before)
