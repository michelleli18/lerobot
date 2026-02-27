# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Simplified SmolVLM wrapper for SmolVLA-FAST (no action expert)."""

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)


class SmolVLMForFast(nn.Module):
    """SmolVLM wrapper for autoregressive FAST action prediction.

    Unlike SmolVLMWithExpertModel, this class does NOT have an action expert.
    Instead, it uses the VLM's own lm_head for next-token prediction of action tokens.
    This mirrors how PI0FastPaliGemma wraps PaliGemma without an expert.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        freeze_vision_encoder: bool = True,
        num_vlm_layers: int = -1,
        device: str = "auto",
    ):
        super().__init__()
        if load_vlm_weights:
            print(f"Loading {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)

        self.processor = AutoProcessor.from_pretrained(model_id)

        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]

        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config

        self.freeze_vision_encoder = freeze_vision_encoder
        self._set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    @property
    def hidden_size(self):
        return self.config.text_config.hidden_size

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

    def embed_image(self, image: torch.Tensor):
        """Embed images through vision encoder + connector."""
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=None,
            )
            .last_hidden_state
        )
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        """Embed language/action tokens using the text model's embedding table."""
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool = False,
    ):
        """Forward through SmolVLM's text model (LlamaModel).

        Args:
            attention_mask: 4D attention mask (B, 1, Q, K) with 0.0 for attend and
                large negative for masked positions. Pass pre-computed 4D mask to
                bypass internal causal mask creation.
            position_ids: Position IDs (B, seq_len).
            past_key_values: KV cache (DynamicCache or None).
            inputs_embeds: Input embeddings (B, seq_len, hidden_size).
            use_cache: Whether to return updated KV cache.

        Returns:
            Tuple of (hidden_states, past_key_values).
        """
        outputs = self.get_vlm_model().text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return outputs.last_hidden_state, outputs.past_key_values
