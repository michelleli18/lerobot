#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SmolVLA-FAST: SmolVLA with autoregressive FAST action token prediction.

Replace flow matching action expert with next-token prediction of discrete action tokens. This project heavily references the PI0Fast vs PI0 implementation from LeRobot. 

"""

import logging
import math
from collections import deque
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _scipy_available, _transformers_available

# Conditional import for type checking and lazy loading
if _scipy_available:
    from scipy.fftpack import idct
else:
    idct = None

if _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla_fast.configuration_smolvla_fast import SmolVLAFastConfig
from lerobot.policies.smolvla_fast.smolvlm_for_fast import SmolVLMForFast
from lerobot.policies.utils import (
    populate_queues,
)
from lerobot.utils.constants import (
    ACTION,
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
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


class SmolVLAFastPolicy(PreTrainedPolicy):
    """
    SmolVLA-FAST Policy for LeRobot.
    Uses autoregressive FAST action token prediction instead of flow matching.
    """

    config_class = SmolVLAFastConfig
    name = "smolvla_fast"

    def __init__(
        self,
        config: SmolVLAFastConfig,
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
        try:
            from transformers import AutoProcessor, AutoTokenizer
            
            # Load FAST tokenizer
            self.action_tokenizer = AutoProcessor.from_pretrained(
                config.action_tokenizer_name, trust_remote_code=True
            )

            # Load PaliGemma tokenizer for token conversion
            self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
                config.text_tokenizer_name, trust_remote_code=True, add_eos_token=True, add_bos_token=False
            )
            
            logging.info("Loaded FAST tokenizer for action detokenization")
        except Exception as e:
            logging.error(f"Failed to load FAST tokenizer for action detokenization: {e}")
            logging.error("Tokenizer loading is required for proper policy initialization; aborting.")
            raise RuntimeError("Failed to load required tokenizers for SmolVLAFastPolicy initialization") from e

        # Initialize the core SmolVLAFast model
        self.model = SmolVLAFastPytorch(
            config,
            paligemma_tokenizer=self._paligemma_tokenizer
        )

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        # self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _get_action_chunk(
        self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        # Prepare inputs
        images, img_masks = self.prepare_images(batch)

        # FAST-only mode: use autoregressive decoding
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        # Get decoding parameters
        temperature = self.config.temperature
        max_decoding_steps = self.config.max_decoding_steps

        # Sample action tokens autoregressively
        if self.config.use_kv_cache:
            action_tokens = self.model.sample_actions_fast_kv_cache(
                images, 
                img_masks, 
                tokens, 
                masks,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
            )
        else:
            action_tokens = self.model.sample_actions_fast(
                images, 
                img_masks, 
                tokens, 
                masks,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
            )
        
        # Detokenize action tokens to continuous actions
        action_horizon = self.config.n_action_steps
        action_dim = self.config.output_features[ACTION].shape[0]
        continuous_actions = self.detokenize_actions(
            action_tokens, action_horizon=action_horizon, action_dim=action_dim
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
    def forward(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)

        # Get FAST action tokens from batch
        fast_action_tokens = batch.get(ACTION_TOKENS)  # (B, max_action_tokens)
        fast_action_masks = batch.get(ACTION_TOKEN_MASK)  # (B, max_action_tokens)

        # Use full language tokens (no separation into high_level_task and subtask)
        tokens = batch.get(OBS_LANGUAGE_TOKENS)
        masks = batch.get(OBS_LANGUAGE_ATTENTION_MASK)

        if fast_action_tokens is None or fast_action_masks is None:
            raise ValueError(
                f"SmolVLA-FAST requires {ACTION_TOKENS} and {ACTION_TOKEN_MASK} in the batch"
            )

        loss_dict = self.model.forward(
            images,
            img_masks,
            tokens,
            masks,
            fast_action_tokens,
            fast_action_masks,
        )

        # The inner model computes masked CE loss over FAST action tokens,
        # so padding is already handled via fast_action_masks — no need
        # for additional actions_is_pad or dimension trimming here.
        loss = loss_dict["loss"]
        detailed_loss_dict = {
            "loss": loss.item(),
            "ce_loss": loss_dict["ce_loss"].item(),
        }
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

    def _smolvlm_tokens_to_act_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert SmolVLM tokens to FAST action tokens (inverse of the mapping done in ActionTokenizerProcessorStep).

        Args:
            tokens: SmolVLM token IDs

        Returns:
            Action token IDs
        """
        return self._paligemma_tokenizer.vocab_size - 1 - self.config.fast_skip_tokens - tokens

    def decode_actions_with_fast(
        self, token_ids: list[int], time_horizon: int, action_dim: int, relaxed_decoding: bool = True
    ) -> np.ndarray:
        """
        Decode action token IDs to continuous action values using the FAST tokenizer.

        Args:
            token_ids: List of token IDs to decode.
            time_horizon: The number of timesteps for actions.
            action_dim: The dimensionality of each action.
            relaxed_decoding: Whether to use relaxed decoding (allows partial sequences).

        Returns:
            A numpy array representing the decoded actions.
        """
        decoded_actions = []

        for token in token_ids:
            try:
                decoded_tokens = self.action_tokenizer.bpe_tokenizer.decode(token)
                decoded_dct_coeff = np.array(list(map(ord, decoded_tokens))) + self.action_tokenizer.min_token

                if relaxed_decoding:
                    # expected sequence length
                    expected_seq_len = time_horizon * action_dim
                    diff = expected_seq_len - decoded_dct_coeff.shape[0]

                    # apply truncation if too long
                    if diff < 0:
                        decoded_dct_coeff = decoded_dct_coeff[:expected_seq_len]  # truncate on the right
                    
                    # apply padding if too short
                    elif diff > 0:
                        decoded_dct_coeff = np.pad(
                            decoded_dct_coeff, (0, diff), mode="constant", constant_values=0
                        )

                decoded_dct_coeff = decoded_dct_coeff.reshape(-1, action_dim)
                assert decoded_dct_coeff.shape == (time_horizon, action_dim), (
                    f"Decoded DCT coefficients have shape {decoded_dct_coeff.shape}, "
                    f"expected ({time_horizon}, {action_dim})"
                )

            except Exception as e:
                logging.warning(f"Error decoding tokens: {e}")
                decoded_dct_coeff = np.zeros((time_horizon, action_dim))

            decoded_actions.append(
                idct(decoded_dct_coeff / self.action_tokenizer.scale, axis=0, norm="ortho")
            )

        return np.stack(decoded_actions)

    def detokenize_actions(self, tokens: torch.Tensor, action_horizon: int, action_dim: int) -> torch.Tensor:
        """
        Detokenizes action tokens back to continuous actions.

        This method converts predicted action tokens from the model back to continuous action values
        using the FAST tokenizer. It handles the conversion from PaliGemma token space to action token
        space, then decodes the action tokens to continuous values using DCT decoding. Ie predicted SmolVLM token IDs -> FAST action tokens -> DCT coefficients -> continuous actions.

        Args:
            tokens: The input tensor of tokenized outputs. Shape: (B, seq_len) or (seq_len,)
            action_horizon: The number of timesteps for actions.
            action_dim: The dimensionality of each action.

        Returns:
            The continuous action tensor. Shape: (B, action_horizon, action_dim) or (action_horizon, action_dim)
        """
        if self.action_tokenizer is None or self._paligemma_tokenizer is None:
            raise ValueError(
                "Action tokenizer not initialized. Make sure fast_only=True in config and tokenizers loaded successfully."
            )
        
        # Handle single sample (add batch dimension)
        single_sample = tokens.dim() == 1
        if single_sample:
            tokens = tokens.unsqueeze(0)

        # Convert token IDs to token strings
        decoded_tokens = [self._paligemma_tokenizer.convert_ids_to_tokens(seq.tolist()) for seq in tokens]
        # Get the token sequence for "Action: " to remove it
        action_prefix_ids = self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False)
        action_prefix_tokens = self._paligemma_tokenizer.convert_ids_to_tokens(action_prefix_ids)
        action_prefix_len = len(action_prefix_tokens)

        # Clean tokens by removing everything after the first "|" (end-of-action marker)
        # and removing all occurrences of "Action: " token sequence
        # assert that beginning contain "Action: "
        if self.config.validate_action_token_prefix:
            for token_seq in decoded_tokens:
                assert len(token_seq) >= 2 and token_seq[0] == "Action" and token_seq[1] == ":", (
                    f"Token sequence does not start with ['Action', ':']: {token_seq}"
                )

        cleaned_tokens = []
        for token_seq in decoded_tokens:
            # Remove everything after "|"
            if "|" in token_seq:
                token_seq = token_seq[: token_seq.index("|")]

            # Remove all occurrences of "Action: " token sequence
            i = 0
            while i <= len(token_seq) - action_prefix_len:
                if token_seq[i : i + action_prefix_len] == action_prefix_tokens:
                    # Found a match, remove it
                    token_seq = token_seq[:i] + token_seq[i + action_prefix_len :]
                else:
                    i += 1

            cleaned_tokens.append(token_seq)

        # Convert token strings back to IDs
        raw_action_tokens = [
            torch.tensor(
                self._paligemma_tokenizer.convert_tokens_to_ids(token_seq),
                dtype=torch.long,
                device=tokens.device,
            )
            for token_seq in cleaned_tokens
        ]

        # Convert SmolVLM tokens to FAST action tokens
        action_tokens = [
            self._smolvlm_tokens_to_act_tokens(raw_action_token) for raw_action_token in raw_action_tokens
        ]

        # Decode action tokens to continuous actions
        actions = self.decode_actions_with_fast(
            action_tokens, time_horizon=action_horizon, action_dim=action_dim
        )

        # Convert to tensor and return
        actions_tensor = torch.tensor(actions, dtype=torch.float32, device=tokens.device)

        # Remove batch dimension if input was single sample
        if single_sample:
            actions_tensor = actions_tensor.squeeze(0)

        return actions_tensor


class SmolVLAFastPytorch(nn.Module):
    """Core SmolVLA-FAST model using autoregressive next-token prediction.

    Uses SmolVLM as the backbone and its lm_head for predicting discrete
    action tokens (FAST tokenizer). No action expert is used.
    """

    def __init__(
        self,
        config: SmolVLAFastConfig,
        paligemma_tokenizer: "AutoTokenizer | None" = None,
    ):
        super().__init__()
        self.config = config
        self._paligemma_tokenizer = paligemma_tokenizer

        self.smolvlm = SmolVLMForFast(
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
        logging.info("Enabled gradient checkpointing for SmolVLAFastPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.smolvlm.get_vlm_model().text_model.gradient_checkpointing_disable()
        self.smolvlm.get_vlm_model().vision_model.gradient_checkpointing_disable()
        logging.info("Disabled gradient checkpointing for SmolVLAFastPytorch model")

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

    def embed_prefix_fast(
        self,
        images,
        img_masks,
        tokens,
        masks,
        fast_action_tokens=None,
        fast_action_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Embed images, language tokens, and FAST action tokens.

        Attention pattern:
        - Images + Language: bidirectional among themselves
        - FAST action tokens: attend to images + language, causal among themselves
    
        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens
            masks: Attention masks for tokens
            fast_action_tokens: FAST action tokens (discrete token IDs)
            fast_action_masks: Padding masks for FAST action tokens

        Returns:
            embs: Concatenated embeddings [images, tokens, fast_action_tokens]
            pad_masks: Padding masks
            att_masks: 2D attention mask
            total_t_images: Total number of image tokens
            num_fast_embs: Number of FAST action token embeddings
        """
        embs = []
        pad_masks = []
        att_mask_segments = []
        total_t_images = 0
        num_fast_embs = 0

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

        # Process FAST action tokens (discrete token IDs)
        if fast_action_tokens is not None:

            def fast_action_embed_func(fast_action_tokens):
                fast_emb = self.smolvlm.embed_language_tokens(fast_action_tokens)
                fast_emb_dim = fast_emb.shape[-1]
                return fast_emb * math.sqrt(fast_emb_dim)

            fast_action_emb = self._apply_checkpoint(fast_action_embed_func, fast_action_tokens)
            embs.append(fast_action_emb)

            num_fast_embs = fast_action_tokens.shape[1]
            pad_masks.append(fast_action_masks)
            att_mask_segments.append(("fast", num_fast_embs))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)

        # Create custom 2D attention mask:
        # - Images + Language: bidirectional among themselves
        # - FAST: attend to images + language, causal among themselves
        att_masks = self._create_custom_attention_mask_fast(att_mask_segments, pad_masks, bsize)

        return embs, pad_masks, att_masks, total_t_images, num_fast_embs

    def _create_custom_attention_mask_fast(self, att_mask_segments, pad_masks, bsize):
        """Create custom 2D attention mask.

        Attention rules:
        - Images + Language: bidirectional among themselves
        - FAST: attend to images + language, causal among themselves
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
                    or query_type == "fast"
                    and key_type in ["image", "language"]
                ):
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = True

                # FAST tokens attend causally to themselves
                elif query_type == "fast" and key_type == "fast":
                    fast_len = query_end - query_start
                    causal_mask = torch.tril(torch.ones(fast_len, fast_len, dtype=torch.bool, device=device))
                    att_2d_masks[:, query_start:query_end, key_start:key_end] = causal_mask[None, :, :]

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
        fast_action_tokens,
        fast_action_masks,
    ) -> dict:
        """Forward pass for SmolVLAFAST training.

        This implements the SmolVLAFAST training objective: predict next action token
        using cross-entropy loss.

        Args:
            images: List of image tensors
            img_masks: List of image masks
            tokens: Language instruction tokens
            masks: Attention masks for tokens
            fast_action_tokens: Discrete action token IDs [B, max_action_tokens]
            fast_action_masks: Padding masks for fast action tokens [B, max_action_tokens]

        Returns:
            Dictionary with 'ce_loss' and 'loss' keys
        """
        if fast_action_tokens is None or fast_action_masks is None:
            raise ValueError("fast_action_tokens and fast_action_masks are required for FAST training")

        # Embed prefix with FAST tokens
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, num_fast_embs = (
            self.embed_prefix_fast(
                images,
                img_masks,
                tokens,
                masks,
                fast_action_tokens=fast_action_tokens,
                fast_action_masks=fast_action_masks,
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

        # Get logits for FAST action tokens using the FAST LM head
        # only compute logits for the positions that predict FAST tokens        
        lm_head = self.smolvlm.vlm.lm_head

        # Targets are the FAST action tokens
        fast_targets = fast_action_tokens  # (B, num_fast_embs)

        # extract logits for FAST token prediction
        fast_hidden = prefix_out[:, -fast_targets.shape[1]:, :]
        fast_logits_for_pred = lm_head(fast_hidden)  # (B, num_fast_embs, vocab_size)

        # Shift left for next-step prediction and shift target
        # logits[:, i] predicts targets[:, i+1]
        fast_logits_for_pred = fast_logits_for_pred[:, :-1, :]  # logits[:, i] predicts targets[:, i+1]
        fast_targets = fast_targets[:, 1:]  # shift targets right
        fast_action_masks = fast_action_masks[:, 1:]  # shift masks to match targets

        # compute cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        fast_logits_flat = fast_logits_for_pred.reshape(-1, fast_logits_for_pred.size(-1))
        fast_targets_flat = fast_targets.reshape(-1)

        fast_loss_per_token = loss_fct(fast_logits_flat, fast_targets_flat)
        fast_loss_per_token = fast_loss_per_token.reshape(fast_targets.shape)

        # apply mask and compute mean loss
        masked_fast_loss = fast_loss_per_token * fast_action_masks.float()
        fast_loss = masked_fast_loss.sum() / fast_action_masks.sum().clamp(min=1)

        return {
            "ce_loss": fast_loss,
            "loss": fast_loss,
        }

    @torch.no_grad()
    def sample_actions_fast(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> torch.Tensor:
        """
        Inefficient but safe autoregressive decoding for FAST tokens.
        Matches the pattern of _generate_subtask_tokens.
        """
        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_action_tokens

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.smolvlm.vlm.lm_head

        # add bos token after tokens
        bos_token = torch.full(
            (bsize, 1), self._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        tokens = torch.cat([tokens, bos_token], dim=1)
        masks = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)

        # 1. Initial Embedding (matches training prefix)
        # prefix_embs will include [Images, Language Prompt, BOS]
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = self.embed_prefix_fast(
            images, img_masks, tokens, masks, fast_action_tokens=None, fast_action_masks=None
        )

        if (
            self.smolvlm.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        generated_action_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)

        # 2. Decoding Loop (each step re-computes full sequence)
        for t in range(max_decoding_steps):
            # always re-calculate position IDs from the current pad mask
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
            last_logits = lm_head(prefix_out[:, -1:, :])  # (B, 1, vocab_size)

            if temperature > 0:
                probs = torch.softmax(last_logits[:, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(last_logits[:, -1], dim=-1, keepdim=True)

            generated_action_tokens[:, t] = next_token.squeeze(-1)

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

        return generated_action_tokens

    @torch.no_grad()
    def sample_actions_fast_kv_cache(
        self,
        images,
        img_masks,
        tokens,
        masks,
        max_decoding_steps=None,
        temperature=0.0,
    ) -> torch.Tensor:
        """Optimized autoregressive decoding for FAST tokens using KV cache."""
        if max_decoding_steps is None:
            max_decoding_steps = self.config.max_action_tokens

        bsize = tokens.shape[0]
        device = tokens.device
        lm_head = self.smolvlm.vlm.lm_head

        # --- PREFILL PHASE ---        
        # Process Images + Text Prompt + BOS token once to populate the KV cache.

        # Add BOS token to the prompt
        bos_token = torch.full(
            (bsize, 1), self._paligemma_tokenizer.bos_token_id, dtype=torch.long, device=device
        )
        tokens_in = torch.cat([tokens, bos_token], dim=1)
        masks_in = torch.cat([masks, torch.ones((bsize, 1), dtype=torch.bool, device=device)], dim=1)

        # Embed prefix [Images, Language, BOS]
        # fast_action_tokens=None means we are just embedding the condition (images+text)
        prefix_embs, prefix_pad_masks, prefix_att_masks, total_t_images, _ = self.embed_prefix_fast(
            images, img_masks, tokens_in, masks_in, fast_action_tokens=None, fast_action_masks=None
        )

        # Ensure correct precision (bfloat16/float32)
        if (
            self.smolvlm.get_vlm_model().text_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Create position IDs (cumsum of mask - 1)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Create 4D mask for the prefix (images + language + BOS), no FAST tokens yet
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

        # Initialize storage for generated tokens
        generated_action_tokens = torch.zeros((bsize, max_decoding_steps), dtype=torch.long, device=device)
        generated_action_tokens[:, 0] = next_token.squeeze(-1)


        # Track valid tokens mask (0 for pad, 1 for valid)
        # We need this to tell the new token what it can attend to (images + text + past actions)
        current_pad_mask = prefix_pad_masks

        # --- 2. DECODING PHASE ---
        # Generate remaining tokens one by one using the cache.

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

            generated_action_tokens[:, t] = next_token.squeeze(-1)

        return generated_action_tokens

