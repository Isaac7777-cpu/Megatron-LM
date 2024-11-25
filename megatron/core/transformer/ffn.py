from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.f2py.auxfuncs import isintent_aux

from core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import bias_geglu_impl
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_global_memory_buffer,
)
from functools import partial
from megatron.core.utils import is_torch_min_version

from mlp import MLP, MLPSubmodules

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    'tensor_model_parallel': False,
    'partition_dim': -1,
    'partition_stride': 1,
}


if is_torch_min_version("2.4.0a0"):
    custom_fwd = partial(torch.amp.custom_fwd, device_type="cuda")
    custom_bwd = partial(torch.amp.custom_bwd, device_type="cuda")
else:
    custom_fwd = torch.cuda.amp.custom_fwd
    custom_bwd = torch.cuda.amp.custom_bwd


if is_torch_min_version("1.13.0"):
    dist_all_gather_func = torch.distributed.all_gather_into_tensor
    dist_reduce_scatter_func = torch.distributed.reduce_scatter_tensor
else:
    dist_all_gather_func = torch.distributed._all_gather_base
    dist_reduce_scatter_func = torch.distributed._reduce_scatter_base

class FFNFused(MLP):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: int = None,
    ):
        super().__init__(config, submodules, is_expert=is_expert, input_size=input_size)
        print('[IL_DEBUG]Using custom FFN')
    
    def forward(self, hidden_states):
        print('[IL_DEBUG]Using fused forward')
        """Perform the forward pass through the MLP block."""
        assert isinstance(self.linear_fc1, ColumnParallelLinear), "The first linear mapping should always be `ColumnParallelLinear`"
        assert  isinstance(self.linear_fc2, RowParallelLinear), "The second linear mapping should always be `RowParallelLinear`"
        # ----The start of the first linear transformation----
        # [s, b, 4 * h/p]
        # intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        # This is the first linear layer that takes the linear mapping
        weight1 = self.linear_fc1.weight

        if self.linear_fc1.config._cpu_offloading_context is not None:
            if self.linear_fc1.config._cpu_offloading_context.inside_context is True:
                assert (
                    self.linear_fc1.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias1 = self.linear_fc1.bias if not self.linear_fc1.skip_bias_add else None

        if (
            self.linear_fc1.allreduce_dgrad
            or self.linear_fc1.sequence_parallel
            or self.linear_fc1.explicit_expert_comm
            or self.linear_fc1.disable_grad_reduce
        ):
            input_parallel = hidden_states
        else:
            input_parallel = copy_to_tensor_model_parallel_region(hidden_states)    # I believe this function does not involve saving anything to the context

        if self.linear_fc1.config.defer_embedding_wgrad_compute:
            if (
                self.linear_fc1.wgrad_deferral_limit == 0
                or len(self.linear_fc1.embedding_activation_buffer) < self.linear_fc1.wgrad_deferral_limit
            ):
                self.linear_fc1.embedding_activation_buffer.append(input_parallel)

        if self.linear_fc1.config.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input_parallel.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            dist_all_gather_func(all_gather_buffer, input, group=get_tensor_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input_parallel

        output1 = torch.matmul(total_input, weight1.t())
        if bias1 is not None:
            output1 = output1 + bias1
        # ----End of First Linear Section----

        # I think the default usage of GPT is actually using bias activation fusion.
        # Also, the default spec (which is the one we are using) is using gelu. Check if the goal is to use `silu`?
        # Furthermore, the default spec is not using a gated_linear_unit as well.
        if self.config.bias_activation_fusion:
            if self.activation_func == F.gelu:
                if self.config.gated_linear_unit:
                    raise NotImplementedError('Only support silu with gated')
                else:
                    raise NotImplementedError('Only support silu with gated')
            elif self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    self.config.activation_func_fp8_input_store,
                )
            else:
                raise ValueError("Only support fusion of silu with gated")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if self.config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return self.config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)

        return output, output_bias