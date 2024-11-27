from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_global_memory_buffer,
)
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import copy_to_tensor_model_parallel_region
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    'tensor_model_parallel': False,
    'partition_dim': -1,
    'partition_stride': 1,
}

TORCH_VERSION = torch.__version__.split('.')

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


class FusedFFN(MLP):
    def __init__(
            self,
            config: TransformerConfig,
            submodules: MLPSubmodules,
            is_expert: bool = False,
            input_size: int = None,
    ):
        super().__init__(config, submodules, is_expert=is_expert, input_size=input_size)
        print('\n[IL_DEBUG]Using custom FFN')

        # Add gamma (scaling) and beta (bias) as learnable parameters
        self.gamma = torch.nn.Parameter(torch.ones(config.hidden_size))
        self.beta = torch.nn.Parameter(torch.zeros(config.hidden_size))

    def forward(self, hidden_states):
        return FFNFusedSimple.apply(hidden_states)
        return FFNFusedSimple.apply(hidden_states, self.config, self.linear_fc1, self.linear_fc2, self.gamma, self.beta)


class FFNFusedSimple(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            _input: torch.Tensor,
            config: TransformerConfig,
            linear_fc1: ColumnParallelLinear,
            linear_fc2: RowParallelLinear,
            norm_gamma: Optional[torch.Tensor] = None,
            norm_beta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for fused FFN with normalization, scaling, and bias.

        Args:
            ctx: Context for saving tensors for backward pass.
            _input (Tensor): Input tensor to normalize. Shape: (batch_size, seq_len, hidden_size).
            config (TransformerConfig): Configuration object for the transformer.
            linear_fc1 (ColumnParallelLinear): Up-projection linear layer.
            linear_fc2 (RowParallelLinear): Down-projection linear layer.
            norm_gamma (Tensor, optional): Scaling parameter for normalization. Shape: (hidden_size,).
            norm_beta (Tensor, optional): Bias parameter for normalization. Shape: (hidden_size,).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - The final output tensor with the same shape as `_input`.
                - The bias tensor from `linear_fc2` or `None`.
        """
        # Ensure correct types for linear layers
        assert isinstance(linear_fc1, ColumnParallelLinear), \
            "linear_fc1 must be an instance of ColumnParallelLinear"
        assert isinstance(linear_fc2, RowParallelLinear), \
            "linear_fc2 must be an instance of RowParallelLinear"

        # Save original shape for restoring later
        shape = _input.shape
        _input = _input.contiguous().view(-1, _input.size(-1))  # Flatten leading dimensions

        # Debugging info
        if getattr(config, "debug", False):
            print("Input shape:", shape)
            print("Flattened input shape:", _input.shape)

        # Normalization
        if config.normalization == "LayerNorm":
            mean = _input.mean(dim=-1, keepdim=True)
            var = _input.var(dim=-1, unbiased=False, keepdim=True)
            normalized_input = (_input - mean) / ((var + config.layernorm_epsilon) ** 0.5)

        elif config.normalization == "RMSNorm":
            rms = ((_input ** 2).mean(dim=-1, keepdim=True) + config.layernorm_epsilon) ** 0.5
            normalized_input = _input / rms

        else:
            raise NotImplementedError("Only LayerNorm and RMSNorm are supported.")

        # Apply scaling (gamma) and bias (beta) if provided
        if norm_gamma is not None:
            assert norm_gamma.dim() == 1 and norm_gamma.size(0) == _input.size(-1), \
                "norm_gamma must be a 1D tensor with size matching the hidden dimension"
            normalized_input = normalized_input * norm_gamma[None, :]
            if norm_beta is not None:
                assert norm_beta.dim() == 1 and norm_beta.size(0) == _input.size(-1), \
                    "norm_beta must be a 1D tensor with size matching the hidden dimension"
                normalized_input = normalized_input + norm_beta[None, :]

        # First linear layer (up-projection)
        intermediate = torch.nn.functional.linear(
            normalized_input, linear_fc1.weight, linear_fc1.bias
        )

        # Apply activation and optional gating
        if config.gated_linear_unit:
            def glu(x):
                chunks = torch.chunk(x, 2, dim=-1)
                return config.activation_func(chunks[0]) * chunks[1]

            activated = glu(intermediate)
        else:
            activated = config.activation_func(intermediate)

        # Second linear layer (down-projection)
        output = torch.nn.functional.linear(activated, linear_fc2.weight, linear_fc2.bias)

        # Retrieve the bias from the final layer
        final_bias = linear_fc2.bias

        # Save tensors for backward pass
        ctx.save_for_backward(
            norm_gamma, norm_beta,
            _input, normalized_input,
            mean if config.normalization == "LayerNorm" else None,
            var if config.normalization == "LayerNorm" else rms,
            linear_fc1.weight, linear_fc1.bias,
            intermediate, activated,
            linear_fc2.weight, linear_fc2.bias,
        )

        # Save additional context
        ctx.gated = config.gated_linear_unit
        ctx.shape = shape
        ctx.layernorm_eps = config.layernorm_epsilon
        ctx.normalization = config.normalization
        ctx.act_fn = config.activation_func

        # Restore original shape and return both output and bias
        return output.view(shape), final_bias


class FFNFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, linear_fc1: ColumnParallelLinear, linear_fc2: RowParallelLinear, _input,
                config: TransformerConfig) -> Any:
        raise NotImplemented('This Complex Version is Not Implemented Yet...',
                             'Please use the simple version FFNApplySimple',
                             'The only difference is the simple one only support a specific configs...')

        print('[IL_DEBUG]Using fused forward')
        """Perform the forward pass through the MLP block."""
        assert isinstance(linear_fc1,
                          ColumnParallelLinear), "The first linear mapping should always be `ColumnParallelLinear`"
        assert isinstance(linear_fc2,
                          RowParallelLinear), "The second linear mapping should always be `RowParallelLinear`"

        # ---- Normalisation ----
        # ---- Taken from the `torch_norm` file
        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            version_geq_2_4 = int(TORCH_VERSION[0]) > 2 or (
                    int(TORCH_VERSION[0]) == 2 and int(TORCH_VERSION[1]) >= 4
            )
            assert version_geq_2_4, 'Torch RMSNorm requires PyTorch version >= 2.4.0'

            norm_cls = torch.nn.RMSNorm
        else:
            raise Exception("Only LayerNorm and RMSNorm are currently supported")

        pre_mlp_normaliser = norm_cls(normalized_shape=config.hidden_size, eps=config.layernorm_epsilon)

        norm_input = pre_mlp_normaliser(_input)
        # ---- End of Normalisation ----

        # ----The start of the first linear transformation----
        # [s, b, 4 * h/p]
        # intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
        # This is the first linear layer that takes the linear mapping

        # ---- Excerpts from `ColumnParallelLinear`
        if linear_fc1.weight is None:
            raise RuntimeError(
                "Cannot find the weight supplied to ColumnParallelLinear forward pass"
                "weight was not supplied to ColumnParallelLinear forward pass "
                "and skip_weight_param_allocation is True. (self.weight is None)"
            )
        weight1 = linear_fc1.weight

        if linear_fc1.config._cpu_offloading_context is not None:
            if linear_fc1.config._cpu_offloading_context.inside_context is True:
                assert (
                        linear_fc1.config.cpu_offloading is False
                ), "CPU Offloading cannot be enabled while using non-TE modules"

        bias1 = linear_fc1.bias if not linear_fc1.skip_bias_add else None

        if (
                linear_fc1.allreduce_dgrad
                or linear_fc1.sequence_parallel
                or linear_fc1.explicit_expert_comm
                or linear_fc1.disable_grad_reduce
        ):
            input_parallel = norm_input
        else:
            input_parallel = copy_to_tensor_model_parallel_region(
                norm_input)  # I believe this function does not involve saving anything to the context

        if linear_fc1.config.defer_embedding_wgrad_compute:
            if (
                    linear_fc1.wgrad_deferral_limit == 0
                    or len(linear_fc1.embedding_activation_buffer) < linear_fc1.wgrad_deferral_limit
            ):
                linear_fc1.embedding_activation_buffer.append(input_parallel)

        if weight1.requires_grad:
            if linear_fc1.config.sequence_parallel:
                world_size = get_tensor_model_parallel_world_size()
                dim_size = list(input_parallel.size())
                dim_size[0] = dim_size[0] * world_size

                all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
                dist_all_gather_func(all_gather_buffer, input, group=get_tensor_model_parallel_group())
                total_input = all_gather_buffer
            else:
                total_input = input_parallel

            output_parallel1 = torch.matmul(total_input, weight1.t())
            if bias1 is not None:
                output_parallel1 = output_parallel1 + bias1
        else:
            pass

        intermediate_parallel = output_parallel1
        bias_parallel = bias1 if linear_fc1.skip_bias_add else None
        # return output, output_bias
        # ----End of First Linear Section----

        # ----Activation Level----
        # I think the default usage of GPT is actually using bias activation fusion.
        # Also, the default spec (which is the one we are using) is using gelu. Check if the goal is to use `silu`?
        # Furthermore, the default spec is not using a gated_linear_unit as well.
        if config.bias_activation_fusion:
            if config.activation_func == F.gelu:
                if config.gated_linear_unit:
                    raise NotImplementedError('Only support silu with gated')
                else:
                    raise NotImplementedError('Only support silu with gated')
            elif config.activation_func == F.silu and config.gated_linear_unit:
                intermediate_parallel = bias_swiglu_impl(
                    intermediate_parallel,
                    bias_parallel,
                    config.activation_func_fp8_input_store,
                )
            else:
                raise ValueError("Only support fusion of silu with gated")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            if config.gated_linear_unit:

                def glu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return config.activation_func(x[0]) * x[1]

                intermediate_parallel = glu(intermediate_parallel)
            else:
                intermediate_parallel = linear_fc2.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = linear_fc2(intermediate_parallel)

        # ---- Context Setup ----
        # ---- Section for Normalization ----

        # ---- Section for First Linear Layer ----
        ctx.save_for_backward(linear_fc1.weight1)
        ctx.use_bias = linear_fc1.bias is not None
        ctx.gradient_accumulation_fusion = linear_fc1.gradient_accumulation_fusion
        ctx.allreduce_dgrad = linear_fc1.allreduce_dgrad
        ctx.sequence_parallel = linear_fc1.sequence_parallel
        ctx.wgrad_deferral_limit = linear_fc1.wgrad_deferral_limit
        ctx.grad_output_buffer = linear_fc1.grad_output_buffer

        return output, output_bias

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass
