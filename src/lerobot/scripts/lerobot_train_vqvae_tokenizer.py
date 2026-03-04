"""
Train VQ-VAE tokenizer for action encoding.

This script:
1. Loads action chunks from LeRobotDataset (with episode sampling)
2. Optionally applies delta transforms (relative vs absolute actions)
3. Extracts specified action dimensions for encoding
4. Applies normalization (MEAN_STD, MIN_MAX, QUANTILES, or other modes)
5. Trains VQ-VAE tokenizer on the action chunks
6. Saves tokenizer to output directory
7. Optionally pushes tokenizer to Hugging Face Hub
8. Reports compression statistics

Example:

```shell
python -m lerobot.scripts.lerobot_train_vqvae_tokenizer \
    --repo_id=user/dataset_name \
    --action_horizon=32 \
    --encoded_dims="0:23" \
    --normalization_mode="MEAN_STD" \
    --codebook_size=512 \
    --latent_dim=32 \
    --output_dir="./vqvae_tokenizer_dataset_name"
```
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub import HfApi

from lerobot.configs import parser
from lerobot.configs.types import NormalizationMode
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla_vqvae.vqvae_tokenizer import VQVAETokenizer
from lerobot.scripts.lerobot_train_tokenizer import (
    apply_delta_transform,
    apply_normalization,
    process_episode,
)
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass
class VQVAETokenizerTrainingConfig:
    """Configuration for training VQ-VAE tokenizer."""

    # LeRobot dataset repository ID
    repo_id: str
    # Root directory for dataset (default: ~/.cache/huggingface/lerobot)
    root: str | None = None
    # Number of future actions in each chunk
    action_horizon: int = 32
    # Max episodes to use (None = all episodes in dataset)
    max_episodes: int | None = None
    # Fraction of chunks to sample per episode
    sample_fraction: float = 0.1
    # Comma-separated dimension ranges to encode (e.g., "0:6,7:23")
    encoded_dims: str = "0:23"
    # Comma-separated dimension indices for delta transform (e.g., "0,1,2,3,4,5")
    delta_dims: str | None = None
    # Whether to apply delta transform (relative actions vs absolute actions)
    use_delta_transform: bool = False
    # Dataset key for state observations (default: "observation.state")
    state_key: str = OBS_STATE
    # Normalization mode (MEAN_STD, MIN_MAX, QUANTILES, QUANTILE10, IDENTITY)
    normalization_mode: str = "MEAN_STD"

    # VQ-VAE architecture
    hidden_dim: int = 256
    latent_dim: int = 32
    codebook_size: int = 512
    n_encoder_layers: int = 2
    n_decoder_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    commitment_cost: float = 0.25
    codebook_ema_decay: float = 0.99
    use_ema: bool = True

    # Training
    batch_size: int = 256
    num_epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    log_interval: int = 100
    eval_interval: int = 1000
    val_fraction: float = 0.1
    seed: int = 42
    output_dir: str | None = None
    # Whether to push the tokenizer to Hugging Face Hub
    push_to_hub: bool = False
    # Hub repository ID (e.g., "username/tokenizer-name"). If None, uses output_dir name
    hub_repo_id: str | None = None
    # Whether to create a private repository on the Hub
    hub_private: bool = False
    device: str = "cuda"

    # Wandb
    wandb_enable: bool = True
    wandb_project: str = "smolvla_b1k_vqvae_tokenizer"
    wandb_name: str | None = None


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_action_chunks(cfg: VQVAETokenizerTrainingConfig) -> np.ndarray:
    """Load and preprocess action chunks from dataset.

    Reuses process_episode and apply_normalization from lerobot_train_tokenizer.
    """
    # load dataset
    print(f"Loading dataset: {cfg.repo_id}")
    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root)
    print(f"Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # parse normalization mode
    try:
        norm_mode = NormalizationMode(cfg.normalization_mode)
    except ValueError as err:
        raise ValueError(
            f"Invalid normalization_mode: {cfg.normalization_mode}. "
            f"Must be one of: {', '.join([m.value for m in NormalizationMode])}"
        ) from err
    print(f"Normalization mode: {norm_mode.value}")

    # parse encoded dimensions
    encoded_dim_ranges = []
    for range_str in cfg.encoded_dims.split(","):
        start, end = map(int, range_str.strip().split(":"))
        encoded_dim_ranges.append((start, end))

    total_encoded_dims = sum(end - start for start, end in encoded_dim_ranges)
    print(f"Encoding {total_encoded_dims} dimensions: {cfg.encoded_dims}")

    # parse delta dimensions
    delta_dim_list = None
    if cfg.delta_dims is not None and cfg.delta_dims.strip():
        delta_dim_list = [int(d.strip()) for d in cfg.delta_dims.split(",")]
        print(f"Delta dimensions: {delta_dim_list}")
    else:
        print("No delta dimensions specified")

    print(f"Use delta transform: {cfg.use_delta_transform}")
    if cfg.use_delta_transform and (delta_dim_list is None or len(delta_dim_list) == 0):
        print("Warning: use_delta_transform=True but no delta_dims specified. No delta will be applied.")

    print(f"Action horizon: {cfg.action_horizon}")
    print(f"State key: {cfg.state_key}")

    # determine episodes to process
    num_episodes = dataset.num_episodes
    if cfg.max_episodes is not None:
        num_episodes = min(cfg.max_episodes, num_episodes)

    print(f"Processing {num_episodes} episodes...")

    # process episodes sequentially (to avoid pickling issues with dataset)
    all_chunks = []
    for ep_idx in range(num_episodes):
        if ep_idx % 10 == 0:
            print(f"  Processing episode {ep_idx}/{num_episodes}...")

        chunks = process_episode(
            (
                dataset,
                ep_idx,
                cfg.action_horizon,
                delta_dim_list,
                cfg.sample_fraction,
                cfg.state_key,
                cfg.use_delta_transform,
            )
        )
        if chunks is not None:
            all_chunks.append(chunks)

    # concatenate all chunks
    all_chunks = np.concatenate(all_chunks, axis=0)
    print(f"Collected {len(all_chunks)} action chunks")

    # extract only encoded dimensions FIRST (before normalization)
    encoded_chunks = []
    for start, end in encoded_dim_ranges:
        encoded_chunks.append(all_chunks[:, :, start:end])
    encoded_chunks = np.concatenate(encoded_chunks, axis=-1) # [N, H, D_encoded]
    print(f"Extracted {encoded_chunks.shape[-1]} encoded dimensions")

    # apply normalization to encoded dimensions
    print("\nBefore normalization - overall stats:")
    print(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
    print(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")

    # get normalization stats from dataset
    norm_stats = dataset.meta.stats
    if norm_stats is not None and ACTION in norm_stats:
        action_stats = norm_stats[ACTION]

        # build encoded dimension indices
        encoded_dim_indices = []
        for start, end in encoded_dim_ranges:
            encoded_dim_indices.extend(range(start, end))
        encoded_dim_indices = np.array(encoded_dim_indices)

        # extract stats for encoded dimensions only
        encoded_stats = {}
        for stat_name, stat_values in action_stats.items():
            if isinstance(stat_values, (list, np.ndarray, torch.Tensor)):
                stat_array = np.array(stat_values)
                if len(stat_array) > max(encoded_dim_indices):
                    encoded_stats[stat_name] = stat_array[encoded_dim_indices]

        if encoded_stats:
            print(f"\nNormalization stats for encoded dimensions (mode: {norm_mode.value}):")
            for stat_name, stat_values in encoded_stats.items():
                print(
                    f"  {stat_name}: shape={stat_values.shape}, "
                    f"range=[{np.min(stat_values):.4f}, {np.max(stat_values):.4f}]"
                )

            # apply normalization based on mode
            try:
                encoded_chunks = apply_normalization(encoded_chunks, encoded_stats, norm_mode, eps=1e-8)
                print(f"\nApplied {norm_mode.value} normalization")
            except ValueError as e:
                print(f"Warning: {e}. Using raw actions without normalization.")

            print("\nAfter normalization - overall stats:")
            print(f"  Min: {np.min(encoded_chunks):.4f}, Max: {np.max(encoded_chunks):.4f}")
            print(f"  Mean: {np.mean(encoded_chunks):.4f}, Std: {np.std(encoded_chunks):.4f}")

            print("\nPer-dimension stats (after normalization):")
            for d in range(encoded_chunks.shape[-1]):
                dim_data = encoded_chunks[:, :, d]
                print(
                    f"  Dim {d}: min={np.min(dim_data):7.4f}, max={np.max(dim_data):7.4f}, "
                    f"mean={np.mean(dim_data):7.4f}, std={np.std(dim_data):7.4f}"
                )
        else:
            print("Warning: Could not extract stats for encoded dimensions, using raw actions")
    else:
        print("Warning: No normalization stats found in dataset, using raw actions")

    print(f"Encoded chunks shape: {encoded_chunks.shape}")
    return encoded_chunks


@parser.wrap()
def train_vqvae_tokenizer(cfg: VQVAETokenizerTrainingConfig):
    """
    Train VQ-VAE tokenizer for action encoding.

    Args:
        cfg: VQVAETokenizerTrainingConfig dataclass with all configuration parameters
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # load data
    action_chunks = load_action_chunks(cfg)
    action_dim = action_chunks.shape[-1]

    # train/val split
    n_total = len(action_chunks)
    n_val = max(1, int(n_total * cfg.val_fraction))
    rng = np.random.RandomState(cfg.seed)
    perm = rng.permutation(n_total)
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_data = torch.tensor(action_chunks[train_indices], dtype=torch.float32)
    val_data = torch.tensor(action_chunks[val_indices], dtype=torch.float32)

    print(f"\nTrain set: {len(train_data)} chunks")
    print(f"Val set: {len(val_data)} chunks")

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # create model
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = VQVAETokenizer(
        action_dim=action_dim,
        chunk_size=cfg.action_horizon,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        codebook_size=cfg.codebook_size,
        n_encoder_layers=cfg.n_encoder_layers,
        n_decoder_layers=cfg.n_decoder_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        commitment_cost=cfg.commitment_cost,
        codebook_ema_decay=cfg.codebook_ema_decay,
        use_ema=cfg.use_ema,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Config: {model.get_config()}")

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.num_epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, cfg.warmup_steps, total_steps
    )

    # initialize wandb
    if cfg.wandb_enable:
        import wandb

        wandb_name = cfg.wandb_name or f"vqvae_tok_cb{cfg.codebook_size}_ld{cfg.latent_dim}_ah{cfg.action_horizon}"
        wandb.init(
            project=cfg.wandb_project,
            name=wandb_name,
            config={
                "repo_id": cfg.repo_id,
                "action_horizon": cfg.action_horizon,
                "encoded_dims": cfg.encoded_dims,
                "codebook_size": cfg.codebook_size,
                "latent_dim": cfg.latent_dim,
                "hidden_dim": cfg.hidden_dim,
                "n_encoder_layers": cfg.n_encoder_layers,
                "n_decoder_layers": cfg.n_decoder_layers,
                "n_heads": cfg.n_heads,
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "n_params": n_params,
                "n_train": len(train_data),
                "n_val": len(val_data),
                "action_dim": action_dim,
            },
        )

    # training loop
    global_step = 0
    best_val_loss = float("inf")

    print(f"\nStarting training for {cfg.num_epochs} epochs ({total_steps} steps).")

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_losses = {
            "recon_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
            "total_loss": 0.0,
        }
        epoch_steps = 0

        for (batch_actions,) in train_loader:
            batch_actions = batch_actions.to(device)

            _, indices, total_loss, loss_dict = model(batch_actions)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k].item()
            epoch_steps += 1
            global_step += 1

            if global_step % cfg.log_interval == 0:
                avg_losses = {
                    k: v / epoch_steps for k, v in epoch_losses.items()
                }
                utilization = model.get_codebook_utilization(indices)
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  Step {global_step:6d} | "
                    f"recon: {avg_losses['recon_loss']:.6f} | "
                    f"commit: {avg_losses['commitment_loss']:.6f} | "
                    f"total: {avg_losses['total_loss']:.6f} | "
                    f"util: {utilization:.2%} | "
                    f"lr: {lr:.2e}"
                )
                if cfg.wandb_enable:
                    wandb.log({
                        "train/recon_loss": avg_losses["recon_loss"],
                        "train/commitment_loss": avg_losses["commitment_loss"],
                        "train/codebook_loss": avg_losses["codebook_loss"],
                        "train/total_loss": avg_losses["total_loss"],
                        "train/codebook_utilization": utilization,
                        "train/lr": lr,
                        "train/epoch": epoch,
                    }, step=global_step)

            # periodic evaluation
            if global_step % cfg.eval_interval == 0:
                val_metrics = evaluate(model, val_loader, device)
                print(
                    f"  [EVAL] Step {global_step} | "
                    f"val_recon: {val_metrics['recon_loss']:.6f} | "
                    f"val_total: {val_metrics['total_loss']:.6f} | "
                    f"val_util: {val_metrics['utilization']:.2%}"
                )

                if cfg.wandb_enable:
                    wandb.log({
                        "val/recon_loss": val_metrics["recon_loss"],
                        "val/commitment_loss": val_metrics["commitment_loss"],
                        "val/codebook_loss": val_metrics["codebook_loss"],
                        "val/total_loss": val_metrics["total_loss"],
                        "val/codebook_utilization": val_metrics["utilization"],
                        "val/n_unique_codes": val_metrics["n_unique_codes"],
                    }, step=global_step)

                if val_metrics["total_loss"] < best_val_loss:
                    best_val_loss = val_metrics["total_loss"]
                    output_dir = cfg.output_dir or f"vqvae_tokenizer_{cfg.repo_id.replace('/', '_')}"
                    model.save_pretrained(f"{output_dir}/best")
                    print(f"  Saved best model (val_loss={best_val_loss:.6f})")

                model.train()

        # end of epoch summary
        avg_losses = {k: v / max(epoch_steps, 1) for k, v in epoch_losses.items()}
        print(
            f"\nEpoch {epoch + 1}/{cfg.num_epochs} | "
            f"recon: {avg_losses['recon_loss']:.6f} | "
            f"total: {avg_losses['total_loss']:.6f}"
        )

    # final evaluation
    print("\n" + "=" * 80)
    print("Final evaluation on validation set:")
    val_metrics = evaluate(model, val_loader, device)
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")

    # save final model
    output_dir = cfg.output_dir
    if output_dir is None:
        output_dir = f"vqvae_tokenizer_{cfg.repo_id.replace('/', '_')}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path / "final"))
    print(f"\nSaved final model to {output_path / 'final'}")

    # also copy best as the default checkpoint
    best_path = output_path / "best"
    if best_path.exists():
        import shutil

        for f in best_path.iterdir():
            shutil.copy2(f, output_path / f.name)
        print(f"Copied best checkpoint to {output_path}")

    # save training metadata
    metadata = {
        "repo_id": cfg.repo_id,
        "action_horizon": cfg.action_horizon,
        "encoded_dims": cfg.encoded_dims,
        "normalization_mode": cfg.normalization_mode,
        "model_config": model.get_config(),
        "training_config": {
            "num_epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "total_steps": global_step,
        },
        "final_metrics": {
            k: float(v) for k, v in val_metrics.items() if isinstance(v, float)
        },
    }

    with open(output_path / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # push to Hugging Face Hub if requested
    if cfg.push_to_hub:
        # determine the hub repository ID
        hub_repo_id = cfg.hub_repo_id
        if hub_repo_id is None:
            hub_repo_id = output_path.name
            print(f"\nNo hub_repo_id provided, using: {hub_repo_id}")

        print(f"\nPushing tokenizer to Hugging Face Hub: {hub_repo_id}")
        print(f"   Private: {cfg.hub_private}")

        try:
            api = HfApi()
            api.create_repo(repo_id=hub_repo_id, private=cfg.hub_private, exist_ok=True)
            api.upload_folder(
                folder_path=str(output_path),
                repo_id=hub_repo_id,
                repo_type="model",
                commit_message=f"Upload VQ-VAE tokenizer trained on {cfg.repo_id}",
            )
            print(f"Successfully pushed tokenizer to: https://huggingface.co/{hub_repo_id}")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            print("   Make sure you're logged in with `huggingface-cli login`")

    if cfg.wandb_enable:
        wandb.finish()


@torch.no_grad()
def evaluate(
    model: VQVAETokenizer,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_losses = {
        "recon_loss": 0.0,
        "codebook_loss": 0.0,
        "commitment_loss": 0.0,
        "total_loss": 0.0,
    }
    all_indices = []
    n_batches = 0

    for (batch_actions,) in val_loader:
        batch_actions = batch_actions.to(device)
        _, indices, _, loss_dict = model(batch_actions)

        for k in total_losses:
            total_losses[k] += loss_dict[k].item()
        all_indices.append(indices.cpu())
        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    # codebook utilization across entire val set
    all_indices = torch.cat(all_indices, dim=0)
    utilization = model.get_codebook_utilization(all_indices)
    n_unique = len(all_indices.unique())

    avg_losses["utilization"] = utilization
    avg_losses["n_unique_codes"] = float(n_unique)
    avg_losses["codebook_size"] = float(model.codebook_size)

    return avg_losses

def main():
    """CLI entry point that parses arguments and runs the tokenizer training."""
    train_vqvae_tokenizer()


if __name__ == "__main__":
    main()
