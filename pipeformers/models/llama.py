# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2024 Shukant Pal
# Copyright 2024 HuggingFace Inc. team. All rights reserved.


from flash_attn.bert_padding import unpad_input
from flash_attn import flash_attn_varlen_func
import torch
import torch.functional as F
from torch import nn
from transformers import LlamaConfig
from ..activations import ACT2FN
from ..rope import ROPE_INIT_FUNCTIONS


class LlamaEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, embed_tokens: nn.Embedding):
        super().__init__()
        
        self.config = config
        self.embed_tokens = embed_tokens
        self.rope_type = config.rope_scaling.get("rope_type", "default") if config.rope_scaling else "default"

        if self.rope_type == "dynamic":
            raise ValueError("Dynamic RoPE scaling is not implemented!")

        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config)

    @torch.no_grad()
    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, ...],
    ):
        input_ids, attention_mask = inputs
        position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        self.inv_freq = self.inv_freq.to(input_ids.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = input_ids.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        position_embeddings = torch.stack((cos, sin), dim=0)

        return self.embed_tokens(input_ids), attention_mask, position_embeddings



class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, inputs: tuple[torch.Tensor, ...]):
        if isinstance(inputs, tuple):
            hidden_states = inputs[0]
        else:
            hidden_states = inputs

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class LlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size must be divisible by num_heads"
        assert self.num_key_value_groups * self.num_key_value_heads == self.num_heads, "num_heads must be divisible by num_key_value_heads"

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        hidden_states, attention_mask, position_embeddings = inputs

        cos, sin = position_embeddings

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embedding
        cos = cos.unsqueeze(1).to(query_states.dtype)
        sin = sin.unsqueeze(1).to(query_states.dtype)
        query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        return self._flash_attention(query_states, key_states, value_states, attention_mask, q_len)

    def _flash_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
        q_len: int,
    ):
        bsz = query_states.shape[0]

        query_unpadded_states, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_states, attention_mask)
        key_unpadded_states, indices_k, cu_seqlens_k, max_seqlen_in_batch_k = unpad_input(key_states, attention_mask)
        value_unpadded_states, _, _, _ = unpad_input(value_states, attention_mask)

        output = (flash_attn_varlen_func(
            query_unpadded_states,
            key_unpadded_states,
            value_unpadded_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=self.attention_dropout,
            softmax_scale=None,
            causal=True,
        ).reshape(bsz, q_len, -1).contiguous())
        output = self.o_proj(output)

        return output

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""

        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config=config)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        hidden_states, attention_mask, position_embeddings = inputs

        residual = hidden_states
        hidden_states = self.input_layernorm((hidden_states))
        hidden_states = self.self_attn((hidden_states, attention_mask, position_embeddings))
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attention_mask, position_embeddings

class LlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaConfig
    ):
        super().__init__()

        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._pipeline = [nn.Sequential(
            LlamaEmbedding(self.config, self.embed_tokens),
            *self.layers,
            self.norm,
        )]

    def forward(self, inputs):
        return self._pipeline[0](inputs)


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ):
        super().__init__()

        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._pipeline = [
            nn.Sequential(
                *list(self.model._pipeline[0]),
                self.lm_head,
            )
        ]

    def forward(self, inputs):
        return self._pipeline[0](inputs)