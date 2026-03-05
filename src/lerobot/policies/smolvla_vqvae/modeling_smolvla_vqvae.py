"""SmolVLA + VQ-VAE: SmolVLA with VQ-VAE action tokenization.

Two-stage approach:
  Stage 1: Train VQ-VAE tokenizer standalone (see lerobot_train_vqvae_tokenizer.py)
  Stage 2: Freeze VQ-VAE, train VLA to predict VQ codebook indices autoregressively

Action flow:
  Training: continuous actions -> frozen VQ-VAE encoder -> codebook indices -> VLM cross-entropy loss
  Inference: VLM generates codebook indices autoregressively -> frozen VQ-VAE decoder -> continuous actions
"""

import logging
import math
from collections import deque
from typing import TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla_vqvae.configuration_smolvla_vqvae import SmolVLAVQVAEConfig
from lerobot.policies.smolvla_vqvae.smolvlm_for_discrete_actions import SmolVLMForDiscreteActions
from lerobot.policies.smolvla_vqvae.vqvae_tokenizer import VQVAETokenizer
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    temperature: float | None


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class SmolVLAVQVAEPolicy(PreTrainedPolicy):
    """
    SmolVLA + VQ-VAE Policy.
    Uses a frozen VQ-VAE tokenizer to convert between continuous actions
    and discrete codebook indices, then trains the VLM to predict those
    indices autoregressively.
    """

    config_class = SmolVLAVQVAEConfig
    name = "smolvla_vqvae"

    def __init__(
        self,
        config: SmolVLAVQVAEConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config

        # Load tokenizers first
        from transformers import AutoTokenizer

        self._text_tokenizer = AutoTokenizer.from_pretrained(
            config.text_tokenizer_name,
            trust_remote_code=True,
            add_eos_token=True,
            add_bos_token=False,
        )

        # Load frozen VQ-VAE tokenizer
        if config.vqvae_checkpoint_path:
            self.vqvae_tokenizer = VQVAETokenizer.from_pretrained(
                config.vqvae_checkpoint_path
            )
            if config.device:
                self.vqvae_tokenizer = self.vqvae_tokenizer.to(config.device)
            self.vqvae_tokenizer.eval()
            for p in self.vqvae_tokenizer.parameters():
                p.requires_grad = False
            logging.info(
                f"Loaded frozen VQ-VAE tokenizer from {config.vqvae_checkpoint_path} "
                f"(codebook_size={self.vqvae_tokenizer.codebook_size})"
            )
        else:
            logging.warning(
                "No vqvae_checkpoint_path provided — VQ-VAE tokenizer not loaded. "
                "This is fine if loading from a pretrained policy checkpoint."
            )
            self.vqvae_tokenizer = None

        # Initialize the core SmolVLAVQVAE model
        self.model = SmolVLAVQVAEPytorch(
            config,
            text_tokenizer=self._text_tokenizer
        )

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""        
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Generate actions via autoregressive VQ token generation + VQ-VAE decoding."""
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        # Prepare inputs
        images, img_masks = self.prepare_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # Get decoding parameters
        temperature = self.config.temperature

        # Generate VQ tokens autoregressively
        if self.config.use_kv_cache:
            generated_vlm_tokens = self.model.sample_actions_kv_cache(
                images, 
                img_masks, 
                tokens, 
                masks,
                max_decoding_steps=self.config.chunk_size,
                temperature=temperature,
            )
        else:
            generated_vlm_tokens = self.model.sample_actions(
                images,
                img_masks,
                tokens,
                masks,
                max_decoding_steps=self.config.chunk_size,
                temperature=temperature,
            )

        # Convert VLM tokens back to VQ codebook indices
        vq_indices = self._vlm_tokens_to_vq_indices(generated_vlm_tokens)
        vq_indices = vq_indices.clamp(0, self.config.vqvae_codebook_size - 1)

        # Decode with frozen VQ-VAE
        with torch.no_grad():
            continuous_actions = self.vqvae_tokenizer.decode_from_indices(vq_indices)

        # Pad to max_action_dim if needed
        if continuous_actions.shape[-1] < self.config.max_action_dim:
            continuous_actions = F.pad(
                continuous_actions,
                (0, self.config.max_action_dim - continuous_actions.shape[-1]),
            )

        if self.config.adapt_to_pi_aloha:
            continuous_actions = self._pi_aloha_encode_actions(continuous_actions)

        return continuous_actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0
    def _vq_indices_to_vlm_tokens(self, indices: Tensor) -> Tensor:
        """Map VQ codebook indices [0, C) to VLM token space (top C entries)."""
        return self._text_tokenizer.vocab_size - 1 - indices

    def _vlm_tokens_to_vq_indices(self, vlm_tokens: Tensor) -> Tensor:
        """Inverse mapping: VLM tokens back to VQ codebook indices."""
        return self._text_tokenizer.vocab_size - 1 - vlm_tokens

    def forward(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """Do a full training forward pass to compute the loss.

        Encodes actions with frozen VQ-VAE, then trains VLM with cross-entropy loss.
        Args:
            batch: Training batch containing observations and actions.
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = batch[ACTION]  # (B, T, max_action_dim), normalized

        # Encode actions to VQ tokens using frozen tokenizer
        with torch.no_grad():
            # Slice to actual action_dim and chunk_size
            actual_actions = actions[:, : self.config.chunk_size, : self.config.action_dim]
            vq_indices = self.vqvae_tokenizer.encode_to_indices(actual_actions)  # (B, T)

        # Map VQ codebook indices to VLM token space
        vq_action_tokens = self._vq_indices_to_vlm_tokens(vq_indices)
        vq_action_masks = torch.ones_like(vq_action_tokens, dtype=torch.bool)

        loss_dict = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            vq_action_tokens,
            vq_action_masks,
        )

        loss = loss_dict["loss"]
        detailed_loss_dict = {
            "loss_batch": loss.item(),
            "ce_loss": loss_dict["ce_loss"].item(),
        }

        # pred_ids = loss_dict["logits"].argmax(dim=-1)
        # targets_for_acc = vq_action_tokens[:, 1:pred_ids.shape[1]+1]  # shape (B, T-1)
        # masks_for_acc = vq_action_masks[:, 1:pred_ids.shape[1]+1]      # shape (B, T-1)
        # token_acc = ((pred_ids == targets_for_acc) * masks_for_acc).float().sum() / masks_for_acc.sum().clamp(min=1e-6)
        # detailed_loss_dict["train/token_accuracy"] = token_acc.item()

        return loss, detailed_loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions


class SmolVLAVQVAEPytorch(nn.Module):
    """
    Core SmolVLA+VQVAE model.

    Uses SmolVLM backbone with lm_head for autoregressive prediction of VQ codebook indices. 
    No action expert.
    """

    def __init__(
        self,
        config: SmolVLAVQVAEConfig,
        text_tokenizer=None,
    ):
        super().__init__()
        self.config = config
        self._text_tokenizer = text_tokenizer

        self.smolvlm = SmolVLMForDiscreteActions(
            model_id=config.vlm_model_name,
            freeze_vision_encoder=config.freeze_vision_encoder,
            load_vlm_weights=config.load_vlm_weights,
            num_vlm_layers=config.num_vlm_layers,
            device=config.device if config.device is not None else "auto",
        )

        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.smolvlm.get_vlm_model().text_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.smolvlm.get_vlm_model().vision_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logging.info("Enabled gradient checkpointing for SmolVLAVQVAEPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.smolvlm.get_vlm_model().text_model.gradient_checkpointing_disable()
        self.smolvlm.get_vlm_model().vision_model.gradient_checkpointing_disable()
        logging.info("Disabled gradient checkpointing for SmolVLAVQVAEPytorch model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks, dtype=None):
        """Convert 2D boolean masks to 4D float masks for LlamaModel.

        LlamaModel expects 4D masks (B, 1, Q, K) with 0.0 for attend and
        large negative for masked positions. Passing a pre-computed 4D mask
        bypasses its internal causal mask creation.
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result

    def embed_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        action_tokens=None,
        action_masks=None,
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
        """Embed images, language tokens, and (optionally) action tokens.

        Attention pattern:
        - Images + Language: bidirectional among themselves
        - Action tokens: attend to images + language, causal among themselves
        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens
            masks: Attention masks for tokens
            action_tokens: Action tokens (discrete token IDs)
            action_masks: Padding masks for action tokens
        Returns:
            embs: Concatenated embeddings [images, tokens, fast_action_tokens]
            pad_masks: Padding masks
            att_masks: 2D attention mask
            total_t_images: Total number of image tokens
            num_fast_embs: Number of FAST action token embeddings        """
        embs = []
        pad_masks = []
        att_mask_segments = []
        total_t_images = 0
        num_action_embs = 0

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.smolvlm.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img) 
            # Scale image embeddings by sqrt(dim), matching SmolVLA convention
            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_mask_segments.append(("image", num_img_embs))
            total_t_images += num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.smolvlm.embed_language_tokens(tokens)
            # Normalize language embeddings
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        num_lang_embs = lang_emb.shape[1]
        att_mask_segments.append(("language", num_lang_embs))
        # Process action tokens (VQ codebook indices mapped to VLM token space)
        if action_tokens is not None:

            def action_embed_func(action_tokens):
                action_emb = self.smolvlm.embed_language_tokens(action_tokens)
                action_emb_dim = action_emb.shape[-1]
                return action_emb * math.sqrt(action_emb_dim)
            action_emb = self._apply_checkpoint(action_embed_func, action_tokens)
            embs.append(action_emb)

            num_action_embs = action_tokens.shape[1]
            pad_masks.append(action_masks)
            att_mask_segments.append(("action", num_action_embs))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        # Create custom 2D attention mask:
        # - Images + Language: bidirectional among themselves
        # - actions: attend to images + language, causal among themselves
        att_masks = self._create_attention_mask(att_mask_segments, pad_masks, bsize)

        return embs, pad_masks, att_masks, total_t_images, num_action_embs

    def _create_attention_mask(self, att_mask_segments, pad_masks, bsize):
        """Create custom 2D attention mask.

        Attention rules:
        - Images + Language: bidirectional among themselves
        - Action tokens: attend to images + language, causal among themselves
        """
        total_len = sum(length for _, length in att_mask_segments)
        device = pad_masks.device

        att_2d_masks = torch.zeros(bsize, total_len, total_len, dtype=torch.bool, device=device)

        positions = []
        current_pos = 0
        for seg_type, seg_len in att_mask_segments:
            positions.append((seg_type, current_pos, current_pos + seg_len))
            current_pos += seg_len

        for _i, (query_type, query_start, query_end) in enumerate(positions):
            for _j, (key_type, key_start, key_end) in enumerate(positions):
                # Images and Language can attend to each other bidirectionally
                if (
                    query_type in ["image", "language"]
                    and key_type in ["image", "language"]
                    or query_type == "action"
                    and key_type in ["image", "language"]
                ):
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True
                # action tokens attend causally to themselves
                elif query_type == "action" and key_type == "action":
                    action_len = query_end - query_start
                    causal_mask = torch.tril(torch.ones(action_len, action_len, dtype=torch.bool, device=device))
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None]

        # Apply padding masks
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        att_2d_masks = att_2d_masks & pad_2d_masks

        return att_2d_masks

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        vq_action_tokens,
        vq_action_masks,
    ) -> dict:
        """
        Forward pass for SmolVLAVQVAE training.

        This implements the SmolVLAVQVAE training objective: predict next VQVQE action token
        using cross-entropy loss.

        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens [B, T_text]
            masks: Language attention masks [B, T_text]
            vq_action_tokens: VQ codebook indices mapped to VLM token space [B, chunk_size]
            vq_action_masks: Masks for action tokens [B, chunk_size] — all-ones for VQ-VAE

        Returns:
            Dictionary with 'ce_loss' and 'loss' keys
        """
        # Embed prefix
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, num_action_embs = (
            self.embed_prefix(
                images,
                img_masks,
                tokens,
                masks,
                action_tokens=vq_action_tokens,
                action_masks=vq_action_masks,
            )
        )

        # Convert embeddings to bfloat16 if needed
        if (
            self.smolvlm.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # for next-token prediction, input tokens [0:T-1] to predict tokens [1:T]
        input_embs = prefix_embs
        input_pad_masks = prefix_pad_masks
        input_att_masks = prefix_att_masks

        position_ids = torch.cumsum(input_pad_masks, dim=1) - 1
        att_2d_4d = self._prepare_attention_masks_4d(input_att_masks, dtype=input_embs.dtype)

        # forward pass through SmolVLM text model
        prefix_out, _ = self.smolvlm.forward(
            attention_mask=att_2d_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=input_embs,
            use_cache=False,
        )

        # Get logits for action token positions
        lm_head = self.smolvlm.vlm.lm_head
        action_hidden = prefix_out[:, -vq_action_tokens.shape[1]:, :]
        action_logits_for_pred = lm_head(action_hidden)  # (B, chunk_size, vocab_size)

        # Shift left for next-token prediction ad shift target
        # logits[:, i] predicts targets[:, i+1]
        action_logits_for_pred = action_logits_for_pred[:, :-1, :]  # logits[:, i] predicts targets[:, i+1]
        targets = vq_action_tokens[:, 1:]  # shift targets right
        target_masks = vq_action_masks[:, 1:] # shift masks to match targets

        # compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        logits_flat = action_logits_for_pred.reshape(-1, action_logits_for_pred.size(-1))
        targets_flat = targets.reshape(-1)

        loss_per_token = loss_fct(logits_flat, targets_flat).reshape(targets.shape)
        # apply mask and compute mean loss
        masked_loss = loss_per_token * target_masks.float()
        ce_loss = masked_loss.sum() / target_masks.sum().clamp(min=1)
        # if sum is effectively zero (no valid tokens), return a fallback loss of zero
        if target_masks.sum().item() == 0:
            ce_loss = torch.tensor(0.0, device=ce_loss.device)

        return {
            "ce_loss": ce_loss,
            "loss": ce_loss,
            "logits": action_logits_for_pred,
        }

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> Tensor:
        """
        Inefficient but safe autoregressive decoding for action tokens.
        Matches the pattern of _generate_subtask_tokens.
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.chunk_size

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.smolvlm.vlm.lm_head

        # 1. Initial Embedding (matches training prefix)
        # prefix_embs will include [Images, Language Prompt, BOS]
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = self.embed_prefix(
            images, img_masks, tokens, masks,
            action_tokens=None,
            action_masks=None,
        )

        if (
            self.smolvlm.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        generated_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)

        # 2. Decoding Loop (each step re-computes full sequence)
        for t in range(max_decoding_steps):
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            att_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

            # full forward pass (no kv cache)
            prefix_out, _ = self.smolvlm.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=prefix_embs,
                use_cache=False,
            )

            # predict next token from the very last sequence position
            last_logits = lm_head(prefix_out[:, -1:, :])  # (B, 1, vocab_size)?

            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_tokens[:, t] = next_token.squeeze(-1)

            # 3. Update sequence for next iteration (unless it's the last step)
            if t < max_decoding_steps - 1:
                # embed the newly generated token
                next_token_emb = self.smolvlm.embed_language_tokens(next_token)
                next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
                if prefix_embs.dtype == torch.bfloat16:
                    next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

                # append to embeddings
                prefix_embs = torch.cat([prefix_embs, next_token_emb], dim=1)

                # update padding mask (new token is always valid/1)
                prefix_pad_masks = torch.cat(
                    [prefix_pad_masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1
                )

                # update 2d attention mask: grow the matrix
                old_len = prefix_att_masks.shape[1]
                new_len = old_len + 1
                new_att_masks = torch.zeros((bsize, new_len, new_len), dtype=torch.bool, device=device)
                new_att_masks[:, :old_len, :old_len] = prefix_att_masks
                # new token attends to all non-padding tokens in the updated sequence
                new_att_masks[:, -1, :] = prefix_pad_masks
                prefix_att_masks = new_att_masks

        return generated_tokens

    @torch.no_grad()
    def sample_actions_kv_cache(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> Tensor:
        """Optimized autoregressive decoding for action tokens using KV cache."""
        if max_decoding_steps is None:
            max_decoding_steps = self.config.chunk_size

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.smolvlm.vlm.lm_head

        # --- PREFILL PHASE ---
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _ = self.embed_prefix(
            images, img_masks, tokens, masks,
            action_tokens=None, action_masks=None,
        )

        # Ensure correct precision (bfloat16/float32)
        if (
            self.smolvlm.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Create position IDs (cumsum of mask - 1)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        att_4d = self._prepare_attention_masks_4d(prefix_att_masks, dtype=prefix_embs.dtype)

        # Forward pass (Prefill) with use_cache=True
        prefix_out, past_key_values = self.smolvlm.forward(
            attention_mask=att_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=prefix_embs,
            use_cache=True,
        )

        # Sample the first action token from the last logit of the prefix
        last_logits = lm_head(prefix_out[:, -1:, :])  # (B, 1, V)
        if temperature > 0:
            probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

        generated_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)
        generated_tokens[:, 0] = next_token.squeeze(-1)

        # Track valid tokens mask (0 for pad, 1 for valid)
        # We need this to tell the new token what it can attend to (images + text + past actions)
        current_pad_mask = prefix_pad_masks

        # --- 2. DECODING PHASE ---
        for t in range(1, max_decoding_steps):
            # Embed the single previous token
            # We use embed_language_tokens directly to avoid overhead of full prefix embedding
            next_token_emb = self.smolvlm.embed_language_tokens(next_token)
            next_token_emb = next_token_emb * math.sqrt(next_token_emb.shape[-1])
            if prefix_embs.dtype == torch.bfloat16:
                next_token_emb = next_token_emb.to(dtype=torch.bfloat16)

            # Update Pad Mask: append 1s for the new valid token
            new_column = torch.ones((bsize, 1), dtype=torch.bool, device=device)
            current_pad_mask = torch.cat([current_pad_mask, new_column], dim=1)

            # Update Position IDs for the single new token
            current_position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()

            # Create Attention Mask for the single new step
            # The new token attends to all valid tokens in history (captured by current_pad_mask).
            # Shape becomes (B, 1, 1, Total_Len) which works with HF's cache logic.
            step_att_mask = self._prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1), dtype=next_token_emb.dtype
            )
            # Forward pass (Decoding step)
            # input_embeds is just the new token (B, 1, D)
            step_out, past_key_values = self.smolvlm.forward(
                attention_mask=step_att_mask,
                position_ids=current_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=next_token_emb,
                use_cache=True,
            )

            # Sample next token
            last_logits = lm_head(step_out[:, -1:, :])
            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_tokens[:, t] = next_token.squeeze(-1)

        return generated_tokens

