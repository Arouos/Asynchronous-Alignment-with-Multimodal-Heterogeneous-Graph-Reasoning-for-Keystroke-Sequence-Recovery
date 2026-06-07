# Evaluation entry point.
# Author: Haoyu Qiang.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.dataloader_factory import build_dataloaders, build_dataset
from src.evaluation.evaluator import Evaluator
from src.models.registry import build_model
from src.utils.config_loader import load_experiment_config
from src.utils.device import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_experiment_config(
        default_path=args.default_config,
        model_path=args.model_config,
        dataset_path=args.dataset_config,
        experiment_path=args.experiment_config,
    )
    device = resolve_device(config.get("device"))
    dataset = build_dataset(
        dataset_cfg=config["dataset"],
        cache_dir=config.get("paths", {}).get("cache_root", "."),
        password_digits=int(config.get("experiment", {}).get("password_digits", 6)),
    )
    loaders = build_dataloaders(
        dataset=dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 8)),
        num_workers=int(config.get("num_workers", 0)),
    )
    model = build_model(config["model"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    metrics = Evaluator(model, device).evaluate(loaders["test"])
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
