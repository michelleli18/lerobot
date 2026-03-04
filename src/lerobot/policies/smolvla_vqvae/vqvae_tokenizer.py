"""
VQ-VAE Tokenizer for action chunk tokenization.

Encondes continuous action chunks (T, D) into T discrete codebook indices
and decodes them back.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAETokenizer(nn.Module):
    """
    VQ-VAE for action chunk tokenization.

    Architecture (similar to OAT and QueST):
      Encoder: Linear(D->hidden) -> pos_embed -> 2-layer TransformerEncoder -> Linear(hidden->latent)
      VQ Bottleneck: codebook of size C with vectors of dim latent_dim, EMA updates
      Decoder: Linear(latent->hidden) -> pos_embed -> 2-layer TransformerEncoder -> Linear(hidden->D)

    Input: (B, T, D) action chunks
    Output: T discrete codebook indices per sample
    """

    def __init__(
        self,
        action_dim: int = 23,
        chunk_size: int = 32,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        codebook_size: int = 512,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        commitment_cost: float = 0.25,
        codebook_ema_decay: float = 0.99,
        use_ema: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.n_heads = n_heads
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        self.codebook_ema_decay = codebook_ema_decay

        # Encoder
        self.encoder_input_proj = nn.Linear(action_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers
        )
        self.encoder_output_proj = nn.Linear(hidden_dim, latent_dim)

        # Learned positional embeddings
        self.encoder_pos_embed = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )

        # VQ Codebook
        self.codebook = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(
            self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size
        )

        # EMA codebook update
        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer(
                "ema_embed_sum", self.codebook.weight.data.clone()
            )
            # Track usage for codebook reset
            self.register_buffer("codebook_usage_count", torch.zeros(codebook_size))
            self.codebook_reset_threshold = 1.0  # min avg assignments to stay alive
            self.codebook_reset_interval = 100  # check every N steps
            self._steps_since_reset_check = 0

        # Decoder (self-attention, mirroring encoder structure)
        self.decoder_input_proj = nn.Linear(latent_dim, hidden_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=n_decoder_layers
        )
        self.decoder_output_proj = nn.Linear(hidden_dim, action_dim)

        self.decoder_pos_embed = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions to latent vectors. (B, T, D) -> (B, T, latent_dim)"""
        x = self.encoder_input_proj(actions)
        x = x + self.encoder_pos_embed[:, : actions.shape[1], :]
        x = self.encoder_transformer(x)
        z_e = self.encoder_output_proj(x)
        return z_e

    def quantize(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vector quantization with nearest-neighbor lookup.

        Returns:
            z_q_st: quantized latents with straight-through gradient (B, T, latent_dim)
            indices: codebook indices (B, T)
            codebook_loss: loss to move encoder outputs towards codebook
            commitment_loss: loss to move codebook towards encoder outputs
        """
        B, T, D = z_e.shape
        z_flat = z_e.reshape(-1, D)

        # Nearest neighbor lookup: ||z - e||^2 = ||z||^2 - 2*z.e + ||e||^2
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1)
        z_q = self.codebook(indices).reshape(B, T, D)

        # EMA codebook update
        if self.use_ema and self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).float()
                self.ema_cluster_size.mul_(self.codebook_ema_decay).add_(
                    encodings.sum(0), alpha=1 - self.codebook_ema_decay
                )
                embed_sum = z_flat.t() @ encodings
                self.ema_embed_sum.mul_(self.codebook_ema_decay).add_(
                    embed_sum.t(), alpha=1 - self.codebook_ema_decay
                )
                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + 1e-5)
                    / (n + self.codebook_size * 1e-5)
                    * n
                )
                self.codebook.weight.data.copy_(
                    self.ema_embed_sum / cluster_size.unsqueeze(1)
                )

                # Codebook reset: replace dead codes with random encoder outputs
                self.codebook_usage_count.add_(encodings.sum(0))
                self._steps_since_reset_check += 1
                if self._steps_since_reset_check >= self.codebook_reset_interval:
                    avg_usage = self.codebook_usage_count / self._steps_since_reset_check
                    dead_mask = avg_usage < self.codebook_reset_threshold
                    n_dead = dead_mask.sum().item()
                    if n_dead > 0:
                        # Replace dead codes with random encoder outputs
                        rand_indices = torch.randint(0, z_flat.shape[0], (n_dead,), device=z_flat.device)
                        self.codebook.weight.data[dead_mask] = z_flat[rand_indices].detach()
                        self.ema_embed_sum[dead_mask] = z_flat[rand_indices].detach()
                        self.ema_cluster_size[dead_mask] = 1.0
                    self.codebook_usage_count.zero_()
                    self._steps_since_reset_check = 0

            codebook_loss = torch.tensor(0.0, device=z_e.device)
        else:
            codebook_loss = F.mse_loss(z_e.detach(), z_q)

        commitment_loss = F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        indices = indices.reshape(B, T)
        return z_q_st, indices, codebook_loss, commitment_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode quantized latents back to actions. (B, T, latent_dim) -> (B, T, D)"""
        x = self.decoder_input_proj(z_q)
        x = x + self.decoder_pos_embed[:, : z_q.shape[1], :]
        x = self.decoder_transformer(x)
        actions_hat = self.decoder_output_proj(x)
        return actions_hat

    def forward(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Full forward pass: encode -> quantize -> decode.

        Args:
            actions: (B, T, D) continuous action chunks

        Returns:
            actions_hat: (B, T, D) reconstructed actions
            indices: (B, T) codebook indices
            total_loss: scalar loss
            loss_dict: dict with individual loss components
        """
        z_e = self.encode(actions)
        z_q, indices, codebook_loss, commitment_loss = self.quantize(z_e)
        actions_hat = self.decode(z_q)

        recon_loss = F.mse_loss(actions_hat, actions)
        total_loss = (
            recon_loss + codebook_loss + self.commitment_cost * commitment_loss
        )

        loss_dict = {
            "recon_loss": recon_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "total_loss": total_loss,
        }
        return actions_hat, indices, total_loss, loss_dict

    @torch.no_grad()
    def encode_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions to discrete codebook indices. (B, T, D) -> (B, T)"""
        z_e = self.encode(actions)
        B, T, D = z_e.shape
        z_flat = z_e.reshape(-1, D)
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()
        )
        indices = distances.argmin(dim=1).reshape(B, T)
        return indices

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices back to continuous actions. (B, T) -> (B, T, D)"""
        z_q = self.codebook(indices)
        return self.decode(z_q)

    @torch.no_grad()
    def get_codebook_utilization(self, indices: torch.Tensor) -> float:
        """Compute fraction of codebook entries used in the given indices."""
        unique = indices.unique()
        return len(unique) / self.codebook_size

    def trainable_parameters(self):
        """Return parameters that should be optimized (excludes codebook when using EMA)."""
        for name, param in self.named_parameters():
            if self.use_ema and name == "codebook.weight":
                continue
            yield param

    def get_config(self) -> dict:
        """Return config dict for serialization."""
        return {
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "codebook_size": self.codebook_size,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "n_heads": self.n_heads,
            "commitment_cost": self.commitment_cost,
            "codebook_ema_decay": self.codebook_ema_decay,
            "use_ema": self.use_ema,
        }

    def save_pretrained(self, save_dir: str) -> None:
        """Save model weights and config to directory."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "vqvae_tokenizer.pt")
        with open(save_path / "vqvae_config.json", "w") as f:
            json.dump(self.get_config(), f, indent=2)

    @classmethod
    def from_pretrained(
        cls, load_dir: str, device: str = "cpu"
    ) -> "VQVAETokenizer":
        """Load model from directory."""
        load_path = Path(load_dir)
        with open(load_path / "vqvae_config.json") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = torch.load(
            load_path / "vqvae_tokenizer.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model
