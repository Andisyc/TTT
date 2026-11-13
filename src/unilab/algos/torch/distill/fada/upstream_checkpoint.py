"""Read the author's unchanged model/optimizer payload plus explicit source provenance."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from unilab.algos.torch.distill.fada.checkpoint import (
    LoadedOfficialFADAStudentPolicy,
    _official_source_identity,
)
from unilab.algos.torch.distill.fada.collection_contract import validate_official_fada_preprocessing
from unilab.algos.torch.distill.fada.official_model import (
    build_official_fada_student,
    canonical_official_model_config,
)
from unilab.algos.torch.distill.fada.runtime_policy import OfficialFADAPlaybackProfile


def load_upstream_fada_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu",
    source: Mapping[str, Any] | None = None,
) -> LoadedOfficialFADAStudentPolicy:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _upstream_policy_from_payload(payload, device=device, source=source)


def _upstream_policy_from_payload(
    payload: Any, *, device: str | torch.device,
    source: Mapping[str, Any] | None = None,
) -> LoadedOfficialFADAStudentPolicy:
    """Restore the author's source-bound model from an already loaded CPU payload."""
    if not isinstance(payload, dict) or not isinstance(payload.get("cfg"), dict):
        raise ValueError("expected the original FADA cfg/model_state_dict checkpoint")
    binding = source if source is not None else payload.get("unilab_upstream_source")
    if not isinstance(binding, Mapping) or binding.get("version") != 1:
        raise ValueError("original FADA checkpoint requires an explicit source control/environment binding")
    identity = _official_source_identity(binding["source_identity"])
    OfficialFADAPlaybackProfile.from_mapping(binding["control"], source_identity=identity)
    source_profile = binding.get("source_profile")
    if not isinstance(source_profile, Mapping):
        raise ValueError("original FADA checkpoint is missing the source environment")
    cfg = payload["cfg"]
    defaults = canonical_official_model_config()
    model_config = canonical_official_model_config({key: cfg[key] for key in defaults if key in cfg})
    for key in ("obs_dim", "act_dim", "cmd_dim"):
        if payload.get(key) != model_config[key]:
            raise ValueError(f"original FADA {key} contradicts its model config")
    compact = payload.get("compact_obs")
    if not isinstance(compact, Mapping):
        raise ValueError("original FADA checkpoint is missing compact observation preprocessing")
    preprocessing = validate_official_fada_preprocessing({
        "term_order": compact["term_order"], "term_scale": compact["term_scale"],
        "term_noise": compact["term_noise"], "noise_enabled": compact["noise_enabled"],
    })
    policy: Any = build_official_fada_student(model_config, device="cpu")
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    policy.to(device)
    policy.official_preprocessing = preprocessing
    policy.official_source_identity = identity
    checkpoint = {
        **payload,
        "source_identity": identity, "model_config": model_config,
        "preprocessing": preprocessing, "control_profile": dict(binding["control"]),
        "completed_iterations": int(payload.get("extra", {}).get("num_iters", 0)),
        "runtime_config": {"source_route": "upstream_unilab_oracle", "source_profile": dict(source_profile)},
    }
    return LoadedOfficialFADAStudentPolicy(policy=policy.eval(), checkpoint=checkpoint)
