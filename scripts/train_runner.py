# Training entry point.
# Author: Haoyu Qiang.

from __future__ import annotations

import argparse

import torch

from src.data.dataloader_factory import build_dataloaders, build_dataset
from src.evaluation.evaluator import Evaluator
from src.losses.combined_loss import CombinedLoss
from src.models.registry import build_model
from src.training.trainer import Trainer
from src.utils.config_loader import load_experiment_config
from src.utils.device import resolve_device
from src.utils.seed import set_global_seed


def _loss_closure(loss_module: CombinedLoss, model_name: str):
    def closure(outputs, batch):
        log_probs = outputs.log_probs if hasattr(outputs, "log_probs") else outputs["log_probs"]
        input_lengths = outputs.input_lengths if hasattr(outputs, "input_lengths") else outputs["input_lengths"]
        cross_w = getattr(outputs, "cross_weights_flat", torch.zeros(0, device=log_probs.device))
        cross_src = getattr(outputs, "cross_src_index", torch.zeros(0, dtype=torch.long, device=log_probs.device))
        cross_mask = getattr(outputs, "cross_mask_flat", torch.zeros(0, dtype=torch.bool, device=log_probs.device))
        triangle = getattr(outputs, "triangle_terms", None)
        target_lengths = batch["digit_lengths"].to(log_probs.device)
        return loss_module(
            log_probs=log_probs,
            targets=batch["digits"].to(log_probs.device),
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            cross_modal_weights=cross_w,
            cross_modal_src=cross_src,
            cross_modal_mask=cross_mask,
            num_nodes=max(int(cross_src.max().item()) + 1 if cross_src.numel() else 1, 1),
            triangle_terms=triangle,
        )

    _ = model_name
    return closure


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    config = load_experiment_config(
        default_path=args.default_config,
        model_path=args.model_config,
        dataset_path=args.dataset_config,
        experiment_path=args.experiment_config,
    )
    set_global_seed(args.seed)
    device = resolve_device(config.get("device"))
    dataset_cfg = config["dataset"]
    dataset = build_dataset(
        dataset_cfg=dataset_cfg,
        cache_dir=config.get("paths", {}).get("cache_root", "."),
        password_digits=int(config.get("experiment", {}).get("password_digits", 6)),
    )
    loaders = build_dataloaders(
        dataset=dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 8)),
        num_workers=int(config.get("num_workers", 0)),
        seed=args.seed,
    )
    model = build_model(config["model"]).to(device)
    loss_module = CombinedLoss(blank=len(config["model"].get("symbol_vocab", list(range(10)))))
    evaluator = Evaluator(model, device, blank=len(config["model"].get("symbol_vocab", list(range(10)))))
    trainer = Trainer(
        model=model,
        loss_callable=_loss_closure(loss_module, config["model"]["name"]),
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        cfg=config,
        evaluator=lambda loader: evaluator.evaluate(loader).get("cer", 0.0),
        checkpoint_dir=args.checkpoint_dir,
    )
    summary = trainer.train()
    print(summary)


if __name__ == "__main__":
    main()
