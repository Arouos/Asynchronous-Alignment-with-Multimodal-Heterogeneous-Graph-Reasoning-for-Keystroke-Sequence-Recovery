# Experiment dispatcher.
# Author: Haoyu Qiang.

from __future__ import annotations

import argparse

from src.experiments import (
    AblationRunner,
    CrossDatasetRunner,
    MainComparisonRunner,
    ModalityMissingRunner,
    RobustnessOffsetRunner,
    RobustnessSNRRunner,
)
from src.utils.config_loader import load_yaml


_RUNNERS = {
    "main_comparison": MainComparisonRunner,
    "robustness_snr": RobustnessSNRRunner,
    "robustness_offset": RobustnessOffsetRunner,
    "modality_missing": ModalityMissingRunner,
    "cross_dataset": CrossDatasetRunner,
    "ablation": AblationRunner,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch an experiment runner by name.")
    parser.add_argument("--experiment", required=True, choices=sorted(_RUNNERS))
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    runner_cls = _RUNNERS[args.experiment]
    runner = runner_cls(config)
    print(f"Dispatching runner {runner_cls.__name__}.")
    print("Call ``runner.run`` from a coordinating Python script with the required dataset_configs map.")


if __name__ == "__main__":
    main()
