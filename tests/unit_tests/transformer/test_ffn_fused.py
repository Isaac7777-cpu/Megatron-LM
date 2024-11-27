# This test has taken inspiration from test_mlp

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.ffn import FusedFFN
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestParallelFFN:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=12, num_attention_heads=4, use_cpu_initialization=True
        )
        self.mlp = FusedFFN(transformer_config, get_gpt_layer_local_spec(use_ffn_fused=True).submodules.mlp.submodules)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.mlp, FusedFFN)
        assert issubclass(type(self.mlp), MLP)    #Note that it should also be sub-classing the MLP class

        num_weights = sum([p.numel() for p in self.mlp.parameters()])
        assert num_weights == 1212 + 12 * 2     # Note that the extra 12 x 2 is for the gamma and beta for the affine
                                                # transformation for the normalisation layer that was not included in NLP before.
                                                # Therefore, it is just an addition of 2 hidden_size (=12 as defined in
                                                # the above TransformerConfig.

    def test_cpu_forward(self):
        # [sequence length, micro batch size, hidden size]
        hidden_states = torch.ones((32, 2, self.mlp.config.hidden_size))
        output, output_bias = self.mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == self.mlp.config.hidden_size
        assert output_bias.shape[0] == self.mlp.config.hidden_size
        assert output.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_forward(self):
        mlp = self.mlp
        mlp.cuda()
        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, mlp.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, output_bias = mlp(hidden_states)
        assert output.shape[0] == 32
        assert output.shape[1] == 2
        assert output.shape[2] == mlp.config.hidden_size
        assert output_bias.shape[0] == mlp.config.hidden_size
        assert output.dtype == torch.float32
        assert output.device.type == 'cuda'
        assert output_bias.device.type == 'cuda'
