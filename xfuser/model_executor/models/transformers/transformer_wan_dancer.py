from __future__ import annotations

import types

import torch

from xfuser.model_executor.layers.usp import USP


def _xdit_dancer_self_attention(self, hidden_states, rotary_frequencies):
    from diffsynth.distributed.xdit_context_parallel import rope_apply

    query = self.norm_q(self.q(hidden_states))
    key = self.norm_k(self.k(hidden_states))
    value = self.v(hidden_states)
    query = rope_apply(query, rotary_frequencies, self.num_heads)
    key = rope_apply(key, rotary_frequencies, self.num_heads)
    query = query.unflatten(2, (self.num_heads, -1))
    key = key.unflatten(2, (self.num_heads, -1))
    value = value.unflatten(2, (self.num_heads, -1))
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


def patch_wan_dancer_for_xdit(pipe) -> None:
    for block in pipe.dit.blocks:
        block.self_attn.forward = types.MethodType(
            _xdit_dancer_self_attention,
            block.self_attn,
        )
