from typing import Dict, Optional, Unpack

import torch
from native_sparse_attention_pytorch import SparseAttention
from training.transformers.cache_utils import Cache
from training.transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from training.modeling_flash_qwen3 import Qwen3ForCausalLM, Qwen3Config, Qwen3Attention


class Qwen3AttentionWithNSA(torch.nn.Module):

    def __init__(self, config: Qwen3Config, self_attn: Qwen3Attention, override: Dict[str, int]):
        super().__init__()
        # Setup the sparse attention module with compatible dimensions.
        kwargs = dict()
        kwargs["dim"] = config.hidden_size
        kwargs["dim_head"] = config.head_dim
        kwargs["heads"] = config.num_attention_heads
        kwargs["kv_heads"] = config.num_key_value_heads
        kwargs["causal"] = True
        kwargs.update(override)
        sparse_attention = SparseAttention(**kwargs)
        # Copy the weights from the original attention module to the new sparse attention module.
        to_qkv = sparse_attention.to_qkv
        q_proj = self_attn.q_proj
        k_proj = self_attn.k_proj
        v_proj = self_attn.v_proj
        with torch.no_grad():
            to_qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
        self.sparse_attention = sparse_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_output = self.sparse_attention(hidden_states)
        attn_weights = None
        return attn_output, attn_weights


def replace_attention(
    config: Qwen3Config, module: Qwen3ForCausalLM, override: Dict[str, int]
) -> Qwen3ForCausalLM:
    for layer in module.model.layers:
        assert hasattr(layer, "self_attn")
        assert isinstance(layer.self_attn, Qwen3Attention)
        layer.self_attn = Qwen3AttentionWithNSA(config, layer.self_attn, override)
    return module


override = dict()
override["compress_block_size"] = 4
override["compress_block_sliding_stride"] = 2
override["selection_block_size"] = 4
override["num_selected_blocks"] = 2
override["sliding_window_size"] = 2

config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
module = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", config=config)
model = replace_attention(config, module, override)

model.eval()
model = model.cuda()
input_ids = torch.randint(0, config.vocab_size, (1, 2048)).cuda()
attention_mask = torch.ones_like(input_ids).cuda()
input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
