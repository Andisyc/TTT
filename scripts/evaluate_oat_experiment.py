"""Evaluate committed OAT recordings without simulation or checkpoint loading."""
import json

import hydra
from omegaconf import DictConfig, OmegaConf

from unilab.visualization.experiment_evaluation import evaluate_experiment


@hydra.main(version_base="1.3", config_path="../conf/distill", config_name="experiment_evaluate")
def main(cfg: DictConfig) -> None:
    options = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(options, dict):
        raise ValueError("evaluation configuration must be a mapping")
    print(json.dumps(evaluate_experiment({str(key): value for key, value in options.items()}),
                     ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
