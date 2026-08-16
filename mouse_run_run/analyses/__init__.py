"""Reusable analyses over strict v2 rollout artifacts."""

from mouse_run_run.analyses.cka import run_linear_cka
from mouse_run_run.analyses.intervention import compare_rollouts
from mouse_run_run.analyses.paper_plsc import (
    PaperPLSCConfig,
    PaperPLSCResult,
    fit_paper_plsc,
    load_paper_plsc,
    save_paper_plsc,
)
from mouse_run_run.analyses.plsc import (
    FixedRank,
    FittedPLSC,
    PermutationThreshold,
    PLSCConfig,
    StandardizedSubspace,
    fit_plsc,
    load_fitted_plsc,
    save_fitted_plsc,
)

__all__ = [
    "FixedRank",
    "FittedPLSC",
    "PermutationThreshold",
    "PLSCConfig",
    "PaperPLSCConfig",
    "PaperPLSCResult",
    "StandardizedSubspace",
    "fit_plsc",
    "fit_paper_plsc",
    "compare_rollouts",
    "load_fitted_plsc",
    "load_paper_plsc",
    "run_linear_cka",
    "save_fitted_plsc",
    "save_paper_plsc",
]
