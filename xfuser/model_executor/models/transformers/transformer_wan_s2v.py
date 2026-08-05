from __future__ import annotations

import os
import sys
import types

import torch
import torch.cuda.amp as amp

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from xfuser.model_executor.layers.usp import USP, attention


def import_official_wan():
    first_error = None
    try:
        import wan

        if hasattr(wan, "WanS2V"):
            return wan
        first_error = ImportError("the imported wan package does not expose WanS2V")
    except ImportError as error:
        first_error = error
        repo_path = os.environ.get("WAN22_REPO_PATH")
        if repo_path:
            sys.path.insert(0, repo_path)
            try:
                import wan

                if hasattr(wan, "WanS2V"):
                    return wan
                first_error = ImportError(
                    "the imported wan package does not expose WanS2V"
                )
            except ImportError as error:
                first_error = error
        raise ImportError(
            "Wan2.2-S2V support requires the official Wan2.2 repository. "
            "Install xDiT with the wan-audio extra and set WAN22_REPO_PATH "
            f"to the official checkout. Import failed with: {first_error}"
        ) from first_error


def _xdit_s2v_self_attention(
    self,
    hidden_states,
    sequence_lengths,
    grid_sizes,
    rotary_frequencies,
):
    from wan.modules.s2v.model_s2v import rope_apply

    batch_size, sequence_length = hidden_states.shape[:2]
    query = self.norm_q(self.q(hidden_states)).view(
        batch_size,
        sequence_length,
        self.num_heads,
        self.head_dim,
    )
    key = self.norm_k(self.k(hidden_states)).view(
        batch_size,
        sequence_length,
        self.num_heads,
        self.head_dim,
    )
    value = self.v(hidden_states).view(
        batch_size,
        sequence_length,
        self.num_heads,
        self.head_dim,
    )
    query = rope_apply(query, grid_sizes, rotary_frequencies)
    key = rope_apply(key, grid_sizes, rotary_frequencies)
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
    attention_dtype = value.dtype
    if attention_dtype not in supported_dtypes:
        attention_dtype = torch.bfloat16
    hidden_states = USP(
        query.to(attention_dtype).transpose(1, 2),
        key.to(attention_dtype).transpose(1, 2),
        value.to(attention_dtype).transpose(1, 2),
    ).transpose(1, 2)
    return self.o(hidden_states.flatten(2))


def _xdit_local_attention(
    query,
    key,
    value,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    **kwargs,
):
    supported_dtypes = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
    attention_dtype = value.dtype
    if attention_dtype not in supported_dtypes:
        attention_dtype = torch.bfloat16
    query = query.to(attention_dtype)
    key = key.to(attention_dtype)
    value = value.to(attention_dtype)
    if q_scale is not None:
        query = query * q_scale
    hidden_states = attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dropout_p=dropout_p,
        is_causal=causal,
    ).transpose(1, 2)
    return hidden_states


def patch_s2v_block_for_low_precision(block) -> None:
    bulk_dtype = block.self_attn.q.weight.dtype

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        hidden_states = x
        modulation_inputs = e
        sequence_lengths = seq_lens
        rotary_frequencies = freqs
        context_lengths = context_lens
        segment_index = modulation_inputs[1].item()
        segment_index = min(max(0, segment_index), hidden_states.size(1))
        segments = [0, segment_index, hidden_states.size(1)]
        modulation = self.modulation.unsqueeze(2)
        with amp.autocast(dtype=torch.float32):
            modulation_values = (modulation + modulation_inputs[0]).chunk(
                6,
                dim=1,
            )
        modulation_values = [value.squeeze(1) for value in modulation_values]

        normalized = self.norm1(hidden_states).float()
        attention_inputs = []
        for index in range(2):
            attention_inputs.append(
                normalized[:, segments[index] : segments[index + 1]]
                * (1 + modulation_values[1][:, index : index + 1])
                + modulation_values[0][:, index : index + 1]
            )
        attention_inputs = torch.cat(attention_inputs, dim=1).to(bulk_dtype)
        attention_output = self.self_attn(
            attention_inputs,
            sequence_lengths,
            grid_sizes,
            rotary_frequencies,
        )
        with amp.autocast(dtype=torch.float32):
            gated_attention = []
            for index in range(2):
                gated_attention.append(
                    attention_output[:, segments[index] : segments[index + 1]]
                    * modulation_values[2][:, index : index + 1]
                )
            hidden_states = hidden_states + torch.cat(gated_attention, dim=1)

        cross_input = self.norm3(hidden_states).to(bulk_dtype)
        hidden_states = hidden_states + self.cross_attn(
            cross_input,
            context,
            context_lengths,
        )
        normalized = self.norm2(hidden_states).float()
        feed_forward_inputs = []
        for index in range(2):
            feed_forward_inputs.append(
                normalized[:, segments[index] : segments[index + 1]]
                * (1 + modulation_values[4][:, index : index + 1])
                + modulation_values[3][:, index : index + 1]
            )
        feed_forward_output = self.ffn(
            torch.cat(feed_forward_inputs, dim=1).to(bulk_dtype)
        )
        with amp.autocast(dtype=torch.float32):
            gated_feed_forward = []
            for index in range(2):
                gated_feed_forward.append(
                    feed_forward_output[:, segments[index] : segments[index + 1]]
                    * modulation_values[5][:, index : index + 1]
                )
            hidden_states = hidden_states + torch.cat(
                gated_feed_forward,
                dim=1,
            )
        return hidden_states

    block.forward = types.MethodType(forward, block)


def patch_wan_s2v_for_xdit(transformer) -> None:
    import wan.modules.model as wan_model
    import wan.modules.s2v.model_s2v as model_s2v
    import wan.modules.s2v.motioner as motioner

    model_s2v.get_rank = get_sequence_parallel_rank
    model_s2v.get_world_size = get_sequence_parallel_world_size
    model_s2v.gather_forward = lambda tensor, dim: get_sp_group().all_gather(
        tensor,
        dim=dim,
    )
    wan_model.flash_attention = _xdit_local_attention
    model_s2v.flash_attention = _xdit_local_attention
    motioner.flash_attention = _xdit_local_attention

    for block in transformer.blocks:
        block.self_attn.forward = types.MethodType(
            _xdit_s2v_self_attention,
            block.self_attn,
        )

    transformer.use_context_parallel = get_sequence_parallel_world_size() > 1
