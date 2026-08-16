"""Nature source-data summaries used only as non-gating comparisons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NatureReference:
    category: str
    metric: str
    group: str
    panel: str
    label: str
    unit: str
    paper_mean: float
    paper_sd: float


NATURE_REFERENCES = (
    NatureReference(
        "behavior", "collision", "social", "Fig. 5i", "Chaser collisions / episode", "count", 16.059, 2.6771645448
    ),
    NatureReference(
        "behavior", "collision", "non_social", "Fig. 5i", "Chaser collisions / episode", "count", 3.19, 0.3571180944
    ),
    NatureReference(
        "behavior", "fov", "social", "Fig. 5j", "Chaser partner in vision", "percent", 86.3615, 6.9984764413
    ),
    NatureReference(
        "behavior", "fov", "non_social", "Fig. 5j", "Chaser partner in vision", "percent", 36.161, 1.5491248425
    ),
    NatureReference(
        "behavior", "distance", "social", "Fig. 5k", "Chaser average distance", "distance", 3.03595, 0.4723580092
    ),
    NatureReference(
        "behavior", "distance", "non_social", "Fig. 5k", "Chaser average distance", "distance", 6.442985, 0.1210808799
    ),
    NatureReference(
        "plsc", "significant_dimensions", "social", "Fig. 6g", "Significant shared dimensions", "count", 164.6891655098, 51.9540351823
    ),
    NatureReference(
        "plsc", "significant_dimensions", "non_social", "Fig. 6g", "Significant shared dimensions", "count", 0.015474298, 0.047086182
    ),
    NatureReference(
        "plsc", "significant_dimensions", "non_social_with_vision", "Fig. 6g", "Significant shared dimensions", "count", 0.1672306541, 0.585086316
    ),
    NatureReference(
        "plsc", "delta_pcc", "social", "Fig. 6h", "Chance-subtracted PLSC1 PCC", "correlation", 0.4059917417, 0.0460029333
    ),
    NatureReference(
        "plsc", "delta_pcc", "non_social", "Fig. 6h", "Chance-subtracted PLSC1 PCC", "correlation", -0.0003309429, 0.0270907513
    ),
    NatureReference(
        "plsc", "delta_pcc", "non_social_with_vision", "Fig. 6h", "Chance-subtracted PLSC1 PCC", "correlation", 0.0354233845, 0.0503439428
    ),
    NatureReference(
        "causal", "collision", "baseline", "Fig. 6s", "Collisions / episode", "count", 17.488, 5.7198927729
    ),
    NatureReference(
        "causal", "collision", "shared_readout", "Fig. 6s", "Collisions / episode", "count", 4.548, 3.7620821658
    ),
    NatureReference(
        "causal", "collision", "shifted_random_pc_readout", "Fig. 6s", "Collisions / episode", "count", 16.91, 11.5921017172
    ),
    NatureReference(
        "causal", "fov", "baseline", "Fig. 6t", "Partner in vision", "percent", 89.516, 5.4536658415
    ),
    NatureReference(
        "causal", "fov", "shared_readout", "Fig. 6t", "Partner in vision", "percent", 46.942, 22.1532750526
    ),
    NatureReference(
        "causal", "fov", "shifted_random_pc_readout", "Fig. 6t", "Partner in vision", "percent", 81.892, 11.6469135635
    ),
    NatureReference(
        "causal", "distance", "baseline", "Fig. 6u", "Average distance", "distance", 2.3157428048, 0.246137543
    ),
    NatureReference(
        "causal", "distance", "shared_readout", "Fig. 6u", "Average distance", "distance", 4.7549184268, 1.2433607633
    ),
    NatureReference(
        "causal", "distance", "shifted_random_pc_readout", "Fig. 6u", "Average distance", "distance", 2.6859708365, 0.6154092697
    ),
)
