""" PyTorch TSBeastBase model."""

#
# Copyright (c) 2024 by Contributors for FMFastSim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils      import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "TSBeastBaseConfig"


TSBEASTBASE_PRETRAINED_MODEL_ARCHIVE_LIST = []


TSBEASTBASE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`TSBeastBaseConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TSBEASTBASE_INPUTS_DOCSTRING = r"""
    Args:
        past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
            Context values of the time series. This denotes the history/past time series values.
            For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series, it is
            greater than 1.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers.

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

class TSBeastBaseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`TSBeastBaseModel`]. It is used to instantiate a
    TSBeastBase model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the TSBeastBase {} architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        context_length (`int`, *optional*, defaults to 64)
            The context/history length for the input sequence.
        patch_length (`int`, *optional*, defaults to 8)
            The patch length for the input sequence.
        num_input_channels (`int`):
            Number of input variates. For Univariate, set it to 1.
        patch_stride (`int`, *optional*, defaults to 8):
            Amount of points to stride. If its value is same as patch_length, we get non-overlapping patches.
        d_model (`int`, *optional*, defaults to 16):
            Hidden feature size of the model.
        expansion_factor (`int`, *optional*, defaults to 2):
            Expansion factor to use inside MLP. Recommended range is 2-5. Larger value indicates more complex model.
        num_layers (`int`, *optional*, defaults to 3):
            Number of layers to use. Recommended range is 3-15. Larger value indicates more complex model.
        dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `TSBeastBase` backbone. Recommended range is 0.2-0.7
        mode (`str`, *optional*, defaults to `"common_channel"`):
            Mixer Mode. Determines how to process the channels. Allowed values: "common_channel", "mix_channel". In
            "common_channel" mode, we follow Channel-independent modelling with no explicit channel-mixing. Channel
            mixing happens in an implicit manner via shared weights across channels. (preferred first approach) In
            "mix_channel" mode, we follow explicit channel-mixing in addition to patch and feature mixer. (preferred
            approach when channel correlations are very important to model)
        gated_attn (`bool`, *optional*, defaults to `True`):
            Enable Gated Attention. Preferred Value is True.
        norm_mlp (`str`, *optional*, defaults to `"LayerNorm"`):
            Normalization layer (BatchNorm or LayerNorm).
        self_attn (`bool`, *optional*, defaults to `False`):
            Enable Tiny self attention across patches. This can be enabled when the output of Vanilla TSBeastBase with
            gated attention is not satisfactory. Enabling this leads to explicit pair-wise attention and modelling
            across patches.
        self_attn_heads (`int`, *optional*, defaults to 1):
            Number of self-attention heads. Works only when `self_attn` is set to `True`.
        use_positional_encoding (`bool`, *optional*, defaults to `False`):
            Enable the use of positional embedding for the tiny self-attention layers. Works only when `self_attn` is
            set to `True`.
        positional_encoding_type (`str`, *optional*, defaults to `"sincos"`):
            Positional encodings. Options `"random"` and `"sincos"` are supported. Works only when
            `use_positional_encoding` is set to `True`
        scaling (`string` or `bool`, *optional*, defaults to `"std"`):
            Whether to scale the input targets via "mean" scaler, "std" scaler or no scaler if `None`. If `True`, the
            scaler is set to "mean". This scaling refers to window scaling and doesn't take care of overall data scaling. Irespecive of this scaling, It is mandatory for the user to perform
            standard or minmax scaling on all the features of the data based on their data charecteristics in their data processing module.
        loss (`string`, *optional*, defaults to `"mse"`):
            The loss function for the model corresponding to the `distribution_output` head. For point estimates it is the mean squared
            error "mse".
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal weight initialization distribution.
        post_init (`bool`, *optional*, defaults to `False`):
            Whether to use custom weight initialization from `transformers` library, or the default initialization in
            `PyTorch`. Setting it to `False` performs `PyTorch` weight initialization.
        norm_eps (`float`, *optional*, defaults to 1e-05):
            A value added to the denominator for numerical stability of normalization.
        reconstruction_type (`str`, *optional*, defaults to patchwise):
            Type of reconstruction. Takes value `full` or `patchwise`
        head_dropout (`float`, *optional*, defaults to 0.2):
            The dropout probability the `TSBeastBase` head.
        decoder_num_layers (`int`, *optional*, defaults to 8):
            Number of layers to use in decoder
        decoder_d_model(`int`, *optional*, defaults to 16):
            Defines the hidden feature size of the decoder.
        decoder_mode (`string`, *optional*, defaults to `"common_channel"`):
            Decoder channel mode. Use `"common_channel" for channel-independent modelling and `"mix_channel"` for channel-mixing modelling
        num_patches_layerwise_scale (`list`, *optional*):
            Compression level of the number of patches at each level in backbone.
            For Ex. if we have 3 levels, initial num_patches is 12 - then if num_patches_layerwise_scale is [1,0.5,0.2], then the number of patches
            at each level will be 12,6,2 (i.e scale multiplied by inital num_patches)
        num_channels_layerwise_scale (`list`, *optional*):
            Compression level of the number of channels at each level in backbone.
            For Ex. if we have 3 levels, initial num_channels is 12 - then if num_channels_layerwise_scale is [1,0.5,0.2], then the number of channels
            at each level will be 12,6,2 (i.e scale multiplied by inital num_channels)
        d_model_layerwise_scale (`list`, *optional*):
            Compression level of the d_model at each level in backbone.
            For Ex. if we have 3 levels, initial d_model is 12 - then if d_model_layerwise_scale is [1,0.5,0.2], then the d_model
            at each level will be 12,6,2 (i.e scale multiplied by inital d_model)
        decoder_num_patches_layerwise_scale (`list`, *optional*):
            Expansion level of the number of patches at each level in decoder.
            For Ex. if we have 3 levels, initial num_patches is 12 - then if decoder_num_patches_layerwise_scale is [0.2,0.5,1], then the number of patches
            at each level will be 2,6,12 (i.e scale multiplied by inital num_patches)
        decoder_num_channels_layerwise_scale (`list`, *optional*):
            Expansion level of the number of channels at each level in decoder.
            For Ex. if we have 3 levels, initial num_channels is 12 - then if decoder_num_channels_layerwise_scale is [0.2,0.5,1], then the number of channels
            at each level will be 2,6,12. (i.e scale multiplied by inital num_channels)
        decoder_d_model_layerwise_scale (`list`, *optional*):
            Expansion level of the d_model at each level in decoder.
            For Ex. if we have 3 levels, initial d_model is 12 - then if decoder_d_model_layerwise_scale is [0.2,0.5,1], then the d_model
            at each level will be 2,6,12. (i.e scale multiplied by inital decoder d_model)
        variational (`bool`, *optional*, defaults to `False`):
            When set to True, output of encoder will have mu and log_var embedding and a normal sampled embedding.
        kl_loss_weight (`float`, *optional*, defaults to 0.2):
            Weight to use for KL loss when variational is set to True
    Example:

    ```python
    >>> from transformers import TSBeastBaseConfig, TSBeastBaseModel

    >>> # Initializing a default TSBeastBase configuration
    >>> configuration = TSBeastBaseConfig()

    >>> # Randomly initializing a model (with random weights) from the configuration
    >>> model = TSBeastBaseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "tsbeastbase"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        # Time series specific configuration
        context_length: int = 64,
        patch_length: int = 8,
        num_input_channels: int = 1,
        patch_stride: int = 8,
        # General model configuration
        d_model: int = 16,
        expansion_factor: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        mode: str = "mix_channel",
        gated_attn: bool = True,
        norm_mlp: str = "LayerNorm",
        self_attn: bool = False,
        self_attn_heads: int = 1,
        use_positional_encoding: bool = False,
        positional_encoding_type: str = "sincos",
        scaling: Optional[Union[str, bool]] = "std",
        loss: str = "mse",
        init_std: float = 0.02,
        post_init: bool = False,
        norm_eps: float = 1e-5,
        # General head configuration
        head_dropout: float = 0.2,
        reconstruction_type: str = "patchwise",
        # decoder parameters
        decoder_num_layers: int = 8,
        decoder_d_model: int = 16,
        decoder_mode: str = "mix_channel",
        # mixer configuration
        encoder_resconn: bool = True,
        decoder_resconn: bool = True,
        # compression/expansion level
        num_patches_layerwise_scale: Optional[List[float]] = None,
        num_channels_layerwise_scale: Optional[List[float]] = None,
        d_model_layerwise_scale: Optional[List[float]] = None,
        decoder_num_patches_layerwise_scale: Optional[List[float]] = None,
        decoder_num_channels_layerwise_scale: Optional[List[float]] = None,
        decoder_d_model_layerwise_scale: Optional[List[float]] = None,
        # variational
        variational: bool = False,
        kl_loss_weight: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.decoder_d_model = decoder_d_model
        self.patch_stride = patch_stride
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.patch_length = patch_length
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = mode
        self.gated_attn = gated_attn
        self.norm_mlp = norm_mlp
        self.scaling = scaling
        self.head_dropout = head_dropout

        self.patch_last = True
        self.use_positional_encoding = use_positional_encoding
        self.positional_encoding_type = positional_encoding_type
        self.self_attn = self_attn
        self.self_attn_heads = self_attn_heads
        self.init_std = init_std
        self.post_init = post_init
        self.loss = loss
        self.norm_eps = norm_eps
        self.min_allowed_range = 4
        self.decoder_num_layers = decoder_num_layers
        self.decoder_mode = decoder_mode
        self.reconstruction_type = reconstruction_type

        self.num_patches_layerwise_scale = num_patches_layerwise_scale
        self.num_channels_layerwise_scale = num_channels_layerwise_scale
        self.d_model_layerwise_scale = d_model_layerwise_scale

        self.decoder_d_model_layerwise_scale = decoder_d_model_layerwise_scale
        self.decoder_num_channels_layerwise_scale = decoder_num_channels_layerwise_scale
        self.decoder_num_patches_layerwise_scale = decoder_num_patches_layerwise_scale

        self.encoder_resconn = encoder_resconn
        self.decoder_resconn = decoder_resconn

        self.mixer_resconn = self.encoder_resconn

        self.variational = variational
        self.kl_loss_weight = kl_loss_weight

        self.init_processing = False

    def set_scale(self, scale_param, actual_param, base_param, no_levels):
        if getattr(self, scale_param) is None:
            setattr(self, scale_param, [1] * no_levels)
            setattr(self, actual_param, [getattr(self, base_param)] * no_levels)
        else:
            setattr(
                self, actual_param, [int(getattr(self, base_param) * float(i)) for i in getattr(self, scale_param)]
            )

        for i in getattr(self, actual_param):
            if i < self.min_allowed_range:
                raise ValueError(
                    "Too much compression beyond level %s for param: %s, %s"
                    % (self.min_allowed_range, scale_param, getattr(self, actual_param))
                )

    def check_and_init_preprocessing(self):
        self.init_processing = True

        if not hasattr(self, "num_patches"):
            self.num_patches = (
                max(self.context_length, self.patch_length) - self.patch_length
            ) // self.patch_stride + 1

        self.set_scale(
            scale_param="num_patches_layerwise_scale",
            actual_param="num_patches_layerwise",
            base_param="num_patches",
            no_levels=self.num_layers,
        )

        self.set_scale(
            scale_param="num_channels_layerwise_scale",
            actual_param="num_channels_layerwise",
            base_param="num_input_channels",
            no_levels=self.num_layers,
        )

        self.set_scale(
            scale_param="d_model_layerwise_scale",
            actual_param="d_model_layerwise",
            base_param="d_model",
            no_levels=self.num_layers,
        )

        self.set_scale(
            scale_param="decoder_num_patches_layerwise_scale",
            actual_param="decoder_num_patches_layerwise",
            base_param="num_patches",
            no_levels=self.decoder_num_layers,
        )

        self.set_scale(
            scale_param="decoder_num_channels_layerwise_scale",
            actual_param="decoder_num_channels_layerwise",
            base_param="num_input_channels",
            no_levels=self.decoder_num_layers,
        )

        self.set_scale(
            scale_param="decoder_d_model_layerwise_scale",
            actual_param="decoder_d_model_layerwise",
            base_param="decoder_d_model",
            no_levels=self.decoder_num_layers,
        )

class TSBeastBaseGatedAttention(nn.Module):
    """
    Module that applies gated attention to input data.

    Args:
        in_size (`int`): The input size.
        out_size (`int`): The output size.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__()
        self.attn_layer = nn.Linear(in_size, out_size)
        self.attn_softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        attn_weight = self.attn_softmax(self.attn_layer(inputs))
        inputs = inputs * attn_weight
        return inputs


class TSBeastBaseBatchNorm(nn.Module):
    """
    Compute batch normalization over the sequence length (time) dimension.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length, d_model)`):
                input for Batch norm calculation
        Returns:
            `torch.Tensor` of shape `(batch_size, sequence_length, d_model)`
        """
        output = inputs.transpose(1, 2)  # output: (batch_size, d_model, sequence_length)
        output = self.batchnorm(output)
        return output.transpose(1, 2)


class TSBeastBasePositionalEncoding(nn.Module):
    """
    Class for positional encoding
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()
        # positional encoding: [num_patches x d_model]
        if config.use_positional_encoding:
            self.position_enc = self._init_pe(config)
        else:
            self.position_enc = nn.Parameter(torch.zeros(config.num_patches, config.d_model))

    @staticmethod
    def _init_pe(config: TSBeastBaseConfig) -> nn.Parameter:
        # Positional encoding
        if config.positional_encoding_type == "random":
            position_enc = nn.Parameter(torch.randn(config.num_patches, config.d_model), requires_grad=True)
        elif config.positional_encoding_type == "sincos":
            position_enc = torch.zeros(config.num_patches, config.d_model)
            position = torch.arange(0, config.num_patches).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model))
            position_enc[:, 0::2] = torch.sin(position * div_term)
            position_enc[:, 1::2] = torch.cos(position * div_term)
            position_enc = position_enc - position_enc.mean()
            position_enc = position_enc / (position_enc.std() * 10)
            position_enc = nn.Parameter(position_enc, requires_grad=False)
        else:
            raise ValueError(
                f"{config.positional_encoding_type} is not a valid positional encoder. Available types are 'random' and 'sincos'."
            )
        return position_enc

    def forward(self, patch_input: torch.Tensor):
        # hidden_state: [bs x num_channels x num_patches x d_model]
        hidden_state = patch_input + self.position_enc
        return hidden_state


class TSBeastBaseNormLayer(nn.Module):
    """Normalization block

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        self.norm_mlp = config.norm_mlp

        if "batch" in config.norm_mlp.lower():
            self.norm = TSBeastBaseBatchNorm(config)
        else:
            self.norm = nn.LayerNorm(config.d_model, eps=config.norm_eps)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the normalization layer.
        Returns:
            `torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`
        """
        if "batch" in self.norm_mlp.lower():
            # reshape the data
            inputs_reshaped = torch.reshape(
                inputs,
                (
                    inputs.shape[0] * inputs.shape[1],
                    inputs.shape[2],
                    inputs.shape[3],
                ),
            )  # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]

            # inputs_reshaped: [batch_size*num_channels, num_patches, d_model]
            inputs_reshaped = self.norm(inputs_reshaped)

            # put back data to the original shape
            inputs = torch.reshape(inputs_reshaped, inputs.shape)

        else:
            inputs = self.norm(inputs)

        return inputs


class TSBeastBaseMLP(nn.Module):
    def __init__(self, in_features, out_features, config):
        super().__init__()
        num_hidden = in_features * config.expansion_factor
        self.fc1 = nn.Linear(in_features, num_hidden)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(num_hidden, out_features)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                Input to the MLP layer.
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        inputs = self.dropout1(nn.functional.gelu(self.fc1(inputs)))
        inputs = self.fc2(inputs)
        inputs = self.dropout2(inputs)
        return inputs


class TSBeastBaseChannelFeatureMixerBlock(nn.Module):
    """This module mixes the features in the channel dimension.

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        self.norm = TSBeastBaseNormLayer(config)
        self.gated_attn = config.gated_attn
        self.mlp = TSBeastBaseMLP(
            in_features=config.num_input_channels,
            out_features=config.num_input_channels,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TSBeastBaseGatedAttention(
                in_size=config.num_input_channels, out_size=config.num_input_channels
            )

        self.resconn = config.mixer_resconn

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs (`torch.Tensor` of shape `((batch_size, num_channels, num_patches, d_model))`):
                input to the MLP layer
        Returns:
            `torch.Tensor` of the same shape as `inputs`
        """
        residual = inputs
        inputs = self.norm(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.gated_attn:
            inputs = self.gating_block(inputs)

        inputs = self.mlp(inputs)

        inputs = inputs.permute(0, 3, 2, 1)

        if self.resconn:
            out = inputs + residual
        else:
            out = inputs
        return out


class TSBeastBaseAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[TSBeastBaseConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class PatchMixerBlock(nn.Module):
    """This module mixes the patch dimension.

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        self.norm = TSBeastBaseNormLayer(config)

        self.self_attn = config.self_attn
        self.gated_attn = config.gated_attn

        self.resconn = config.mixer_resconn

        self.mlp = TSBeastBaseMLP(
            in_features=config.num_patches,
            out_features=config.num_patches,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TSBeastBaseGatedAttention(in_size=config.num_patches, out_size=config.num_patches)

        if config.self_attn:
            self.self_attn_layer = TSBeastBaseAttention(
                embed_dim=config.d_model,
                num_heads=config.self_attn_heads,
                dropout=config.dropout,
            )
            self.norm_attn = TSBeastBaseNormLayer(config)

    def forward(self, hidden_state):
        """
        Args:
            hidden_state (`torch.Tensor`): Input tensor.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden_state

        hidden_state = self.norm(hidden_state)

        if self.self_attn:
            batch_size, n_vars, num_patches, d_model = hidden_state.shape
            hidden_state_reshaped = hidden_state.reshape(batch_size * n_vars, num_patches, d_model)

            x_attn, _, _ = self.self_attn_layer(hidden_state_reshaped, output_attentions=False)
            x_attn = x_attn.reshape(batch_size, n_vars, num_patches, d_model)

        # Transpose so that num_patches is the last dimension
        hidden_state = hidden_state.transpose(2, 3)
        hidden_state = self.mlp(hidden_state)

        if self.gated_attn:
            hidden_state = self.gating_block(hidden_state)

        # Transpose back
        hidden_state = hidden_state.transpose(2, 3)

        if self.self_attn:
            hidden_state = self.norm_attn(hidden_state + x_attn)

        if self.resconn:
            out = hidden_state + residual
        else:
            out = hidden_state
        return out


class FeatureMixerBlock(nn.Module):
    """This module mixes the hidden feature dimension.

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        self.norm = TSBeastBaseNormLayer(config)

        self.gated_attn = config.gated_attn

        self.resconn = config.mixer_resconn

        self.mlp = TSBeastBaseMLP(
            in_features=config.d_model,
            out_features=config.d_model,
            config=config,
        )

        if config.gated_attn:
            self.gating_block = TSBeastBaseGatedAttention(in_size=config.d_model, out_size=config.d_model)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.mlp(hidden)

        if self.gated_attn:
            hidden = self.gating_block(hidden)

        if self.resconn:
            out = hidden + residual
        else:
            out = hidden
        return out


class TSBeastBaseLayer(nn.Module):
    """
    The `TSBeastBase` layer that does all three kinds of mixing.

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.

    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        if config.num_patches > 1:
            self.patch_mixer = PatchMixerBlock(config=config)

        self.feature_mixer = FeatureMixerBlock(config=config)

        self.mode = config.mode
        self.num_patches = config.num_patches
        if config.mode == "mix_channel":
            self.channel_feature_mixer = TSBeastBaseChannelFeatureMixerBlock(config=config)

    def forward(self, hidden: torch.Tensor):
        """
        Args:
            hidden (`torch.Tensor` of shape `(batch_size, num_patches, d_model)`):
                Input tensor to the layer.

        Returns:
            `torch.Tensor`: Transformed tensor.
        """
        if self.mode == "mix_channel":
            hidden = self.channel_feature_mixer(hidden)

        if self.num_patches > 1:
            hidden = self.patch_mixer(hidden)
        hidden = self.feature_mixer(hidden)  # hidden: (batch_size x num_patches x d_model)
        return hidden


class LTranspose(nn.Module):
    """Helper module to transpose"""

    def __init__(self, dim1, dim2):
        super(LTranspose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        x = torch.transpose(x, self.dim1, self.dim2)  # Transpose dimensions 1 and 2
        return x


class TSBeastBaseBlock(nn.Module):
    """The main computing framework of the `TSBeastBase` model.

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        num_layers = config.num_layers

        self.mixers = nn.ModuleList()
        current_d_model = config.d_model
        current_num_patches = config.num_patches
        current_num_input_channels = config.num_input_channels

        for i in range(num_layers):
            temp_config = copy.deepcopy(config)
            temp_config.num_patches = config.num_patches_layerwise[i]
            temp_config.num_input_channels = config.num_channels_layerwise[i]
            temp_config.d_model = config.d_model_layerwise[i]

            if current_d_model != temp_config.d_model:
                self.mixers.append(nn.Linear(current_d_model, temp_config.d_model))
                current_d_model = temp_config.d_model

            if current_num_input_channels != temp_config.num_input_channels:
                self.mixers.append(LTranspose(-1, -3))
                self.mixers.append(nn.Linear(current_num_input_channels, temp_config.num_input_channels))
                current_num_input_channels = temp_config.num_input_channels
                self.mixers.append(LTranspose(-1, -3))

            if current_num_patches != temp_config.num_patches:
                self.mixers.append(LTranspose(-1, -2))
                self.mixers.append(nn.Linear(current_num_patches, temp_config.num_patches))
                current_num_patches = temp_config.num_patches
                self.mixers.append(LTranspose(-1, -2))

            self.mixers.append(TSBeastBaseLayer(config=temp_config))

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor`): The input tensor.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        all_hidden_states = []

        embedding = hidden_state

        for mod in self.mixers:
            embedding = mod(embedding)
            if output_hidden_states:
                all_hidden_states.append(embedding)

        if output_hidden_states:
            return embedding, all_hidden_states
        else:
            return embedding, None


class TSBeastBaseDecoder(nn.Module):
    """Decoder for tiny time mixer

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        self.num_input_channels = config.num_input_channels

        decoder_config = copy.deepcopy(config)
        decoder_config.num_layers = config.decoder_num_layers
        decoder_config.d_model = config.d_model_layerwise[-1]
        decoder_config.num_patches = config.num_patches_layerwise[-1]
        decoder_config.num_input_channels = config.num_channels_layerwise[-1]

        decoder_config.dropout = config.head_dropout
        decoder_config.mode = config.decoder_mode

        decoder_config.num_channels_layerwise_scale = config.decoder_num_channels_layerwise_scale
        decoder_config.num_patches_layerwise_scale = config.decoder_num_patches_layerwise_scale
        decoder_config.d_model_layerwise_scale = config.decoder_d_model_layerwise_scale

        decoder_config.num_channels_layerwise = config.decoder_num_channels_layerwise
        decoder_config.num_patches_layerwise = config.decoder_num_patches_layerwise
        decoder_config.d_model_layerwise = config.decoder_d_model_layerwise

        decoder_config.mixer_resconn = config.decoder_resconn

        self.decoder_block = TSBeastBaseBlock(decoder_config)

    def forward(self, hidden_state, output_hidden_states: bool = False):
        """
        Args:
            hidden_state (`torch.Tensor` of shape `(batch_size x nvars x num_patch x d_model)`): The input tensor from backbone.
            output_hidden_states (`bool`, *optional*, defaults to False.):
                Whether to output the hidden states as well.

        Returns:
            `torch.Tensor`: The embedding. `list`: List of all hidden states if `output_hidden_states` is set to
            `True`.
        """
        if output_hidden_states:
            decoder_hidden_states = []
        else:
            decoder_hidden_states = None

        decoder_input = hidden_state

        decoder_output, hidden_states = self.decoder_block(
            hidden_state=decoder_input, output_hidden_states=output_hidden_states
        )  # bs x nvars x n_patches x d_model

        if output_hidden_states:
            decoder_hidden_states.extend(hidden_states)

        return decoder_output, decoder_hidden_states


class TSBeastBaseForReconstructionHead(nn.Module):
    """Reconstruction Head for Reconstruction

    Args:
        config (`TSBeastBaseConfig`, *required*): Configuration.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        self.dropout_layer = nn.Dropout(config.head_dropout)

        head_d_model = config.decoder_d_model_layerwise[-1]

        self.reconstruction_type = config.reconstruction_type

        if config.reconstruction_type == "full":
            self.base_reconstruction_block = nn.Linear((config.num_patches * head_d_model), config.context_length)
        else:
            self.base_reconstruction_block = nn.Linear(head_d_model, config.patch_length)

        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features):
        """
        Args:
            hidden_features `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.
        Returns:
            `torch.Tensor` of shape `(batch_size, context_length, forecast_channels)`.
        """

        if self.reconstruction_type == "full":
            hidden_features = self.flatten(hidden_features)  # [batch_size x n_vars x num_patch * d_model]

        hidden_features = self.dropout_layer(hidden_features)
        reconstruction = self.base_reconstruction_block(
            hidden_features
        )  # [batch_size, n_vars, num_patch, patch_length] or [batch_size x n_vars x context_length]

        if self.reconstruction_type != "full":
            reconstruction = self.flatten(reconstruction)  # [batch_size x n_vars x num_patch*patch_length]

        reconstruction = reconstruction.transpose(-1, -2)  # [batch_size x context_length x n_vars]

        return reconstruction


class TSBeastBasePreTrainedModel(PreTrainedModel):
    # Weight initialization
    config_class = TSBeastBaseConfig
    base_model_prefix = "model"
    main_input_name = "past_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, TSBeastBasePositionalEncoding):
            # initialize positional encoding
            if self.config.positional_encoding_type == "random":
                nn.init.normal_(module.position_enc, mean=0.0, std=0.1)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, TSBeastBaseBatchNorm):
            module.batchnorm.bias.data.zero_()
            module.batchnorm.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                module.bias.data.zero_()


class TSBeastBasePatchify(nn.Module):
    """
    A class to patchify the time series sequence into different patches

    Returns:
        `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()

        self.sequence_length = config.context_length
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride

        if self.sequence_length <= self.patch_length:
            raise ValueError(
                f"Sequence length ({self.sequence_length}) has to be greater than the patch length ({self.patch_length})"
            )

        # get the number of patches
        self.num_patches = (max(self.sequence_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        new_sequence_length = self.patch_length + self.patch_stride * (self.num_patches - 1)
        self.sequence_start = self.sequence_length - new_sequence_length

    def forward(self, past_values: torch.Tensor):
        """
        Parameters:
            past_values (`torch.Tensor` of shape `(batch_size, sequence_length, num_channels)`, *required*):
                Input for patchification

        Returns:
            `torch.Tensor` of shape `(batch_size, num_channels, num_patches, patch_length)`
        """
        sequence_length = past_values.shape[-2]
        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Input sequence length ({sequence_length}) doesn't match model configuration ({self.sequence_length})."
            )
        # output: [bs x new_sequence_length x num_channels]
        output = past_values[:, self.sequence_start :, :]
        # output: [bs x num_patches x num_input_channels x patch_length]
        output = output.unfold(dimension=-2, size=self.patch_length, step=self.patch_stride)
        # output: [bs x num_input_channels x num_patches x patch_length]
        output = output.transpose(-2, -3).contiguous()
        return output


class TSBeastBaseStdScaler(nn.Module):
    """
    Standardize features by calculating the mean and scaling along the first dimension, and then normalizes it by
    subtracting from the mean and dividing by the standard deviation.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-5

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        denominator = observed_indicator.sum(self.dim, keepdim=self.keepdim)
        denominator = denominator.clamp_min(1.0)
        loc = (data * observed_indicator).sum(self.dim, keepdim=self.keepdim) / denominator

        variance = (((data - loc) * observed_indicator) ** 2).sum(self.dim, keepdim=self.keepdim) / denominator
        scale = torch.sqrt(variance + self.minimum_scale)
        return (data - loc) / scale, loc, scale


class TSBeastBaseMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True
        self.minimum_scale = config.minimum_scale if hasattr(config, "minimum_scale") else 1e-10
        self.default_scale = config.default_scale if hasattr(config, "default_scale") else None

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)

        scale = ts_sum / torch.clamp(num_observed, min=1)

        # If `default_scale` is provided, we use it, otherwise we use the scale
        # of the batch.
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)

        # apply default scale where there are no observations
        scale = torch.where(num_observed > 0, scale, default_scale)

        # ensure the scale is at least `self.minimum_scale`
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale

        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)

        return scaled_data, torch.zeros_like(scale), scale


class TSBeastBaseNOPScaler(nn.Module):
    """
    Assigns a scaling factor equal to 1 along the first dimension, and therefore applies no scaling to the input data.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, "scaling_dim") else 1
        self.keepdim = config.keepdim if hasattr(config, "keepdim") else True

    def forward(
        self, data: torch.Tensor, observed_indicator: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        scale = torch.ones_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        loc = torch.zeros_like(data, requires_grad=False).mean(dim=self.dim, keepdim=self.keepdim)
        return data, loc, scale


@dataclass
class TSBeastBaseEncoderOutput(ModelOutput):
    """
    Base class for `TSBeastBaseEncoderOutput`, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TSBeastBaseEncoder(TSBeastBasePreTrainedModel):
    """
    Encoder for TSBeastBase which inputs patched time-series and outputs patched embeddings.

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__(config)
        self.use_return_dict = config.use_return_dict

        self.patcher = nn.Linear(config.patch_length, config.d_model)
        if config.use_positional_encoding:
            self.positional_encoder = TSBeastBasePositionalEncoding(config=config)
        else:
            self.positional_encoder = None
        self.mlp_mixer_encoder = TSBeastBaseBlock(config=config)

        self.d_model = config.d_model
        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @replace_return_docstrings(output_type=TSBeastBaseEncoderOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TSBeastBaseEncoderOutput]:
        r"""
        Args:
            past_values (`torch.FloatTensor` of shape `(batch_size, seq_length, num_input_channels)`):
                Context values of the time series.
                For univariate time series, `num_input_channels` dimension should be 1. For multivariate time series,
                it is greater than 1.

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, n_vars, num_patches, d_model)`
        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # flatten [bs x num_patch x d_model]. common_channel/mix_channel: [bs x n_vars x num_patch x d_model]
        patches = self.patcher(past_values)

        # add positional encoder
        if self.positional_encoder is not None:
            patches = self.positional_encoder(patches)

        last_hidden_state, hidden_states = self.mlp_mixer_encoder(patches, output_hidden_states=output_hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    last_hidden_state,
                    hidden_states,
                ]
            )

        return TSBeastBaseEncoderOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states)


@dataclass
class TSBeastBaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor`  of shape `(batch_size, num_channels, num_patches, d_model)`):
            Hidden-state at the output of the last layer of the model. num_channels, num_patches, d_model refers to the
            parameters of the last layer of the backbone.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        patch_input (`torch.FloatTensor` of shape `(batch_size, num_channels, num_patches, patch_length)`):
            Patched input data to the model.
        loc: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the mean of the context window per channel.
        scale: (`torch.FloatTensor` of shape `(batch_size, 1, num_channels)`,*optional*):
            Gives the std dev of the context window per channel.
        last_hidden_flatten_state (`torch.FloatTensor`  of shape `(batch_size, total_embedding_size)`):
            Flatten form of the last_hidden_state where total_embedding_size = num_channels*num_patches*d_model of the last layer
            in the backbone.
        mu_hidden_flatten_state (`torch.FloatTensor`  of shape `(batch_size, total_embedding_size)`):
            Flatten form of the mean hidden state where total_embedding_size = num_channels*num_patches*d_model of the last layer
            in the backbone.
        log_var_hidden_flatten_state (`torch.FloatTensor`  of shape `(batch_size, total_embedding_size)`):
            Flatten form of the log variance hidden state where total_embedding_size = num_channels*num_patches*d_model of the last layer
            in the backbone.

    """

    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    mu_hidden_flatten_state: torch.FloatTensor = None
    log_var_hidden_flatten_state: torch.FloatTensor = None
    last_hidden_flatten_state: torch.FloatTensor = None
    patch_input: torch.FloatTensor = None
    loc: Optional[torch.FloatTensor] = None
    scale: Optional[torch.FloatTensor] = None


@add_start_docstrings(
    "The TSBeastBase Model for time-series forecasting.",
    TSBEASTBASE_START_DOCSTRING,
)
class TSBeastBaseModel(TSBeastBasePreTrainedModel):
    def __init__(self, config: TSBeastBaseConfig):
        if not config.init_processing:
            config.check_and_init_preprocessing()
        super().__init__(config)

        self.use_return_dict = config.use_return_dict
        self.encoder = TSBeastBaseEncoder(config)
        self.patching = TSBeastBasePatchify(config)

        if config.scaling == "mean":
            self.scaler = TSBeastBaseMeanScaler(config)
        elif config.scaling == "std" or config.scaling is True:
            self.scaler = TSBeastBaseStdScaler(config)
        else:
            self.scaler = TSBeastBaseNOPScaler(config)

        self.d_model = config.d_model

        self.total_embedding_size = (
            config.d_model_layerwise[-1] * config.num_patches_layerwise[-1] * config.num_channels_layerwise[-1]
        )

        self.variational = config.variational

        if self.variational:
            self.mu_linear = nn.Linear(self.total_embedding_size, self.total_embedding_size)
            self.log_var_linear = nn.Linear(self.total_embedding_size, self.total_embedding_size)

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(TSBEASTBASE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TSBeastBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> TSBeastBaseModelOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        if observed_mask is None:
            observed_mask = torch.ones_like(past_values)
        scaled_past_values, loc, scale = self.scaler(past_values, observed_mask)

        patched_x = self.patching(scaled_past_values)  # [batch_size x num_input_channels x num_patch x patch_length

        enc_input = patched_x

        encoder_output = self.encoder(
            enc_input,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(encoder_output, tuple):
            encoder_output = TSBeastBaseEncoderOutput(*encoder_output)

        last_hidden_flatten_state = encoder_output.last_hidden_state.flatten(start_dim=1)

        mu_hidden_flatten_state = None
        log_var_hidden_flatten_state = None

        if output_hidden_states:
            hidden_states = []
            hidden_states.extend(encoder_output.hidden_states)
        else:
            hidden_states = None

        if self.variational is True:
            mu_hidden_flatten_state = self.mu_linear(last_hidden_flatten_state)
            log_var_hidden_flatten_state = self.log_var_linear(last_hidden_flatten_state)
            last_hidden_flatten_state = mu_hidden_flatten_state + torch.exp(
                0.5 * log_var_hidden_flatten_state
            ) * torch.randn_like(mu_hidden_flatten_state)

            if output_hidden_states:
                hidden_states.append(last_hidden_flatten_state)

        if not return_dict:
            return tuple(
                v
                for v in [
                    encoder_output.last_hidden_state,
                    hidden_states,
                    mu_hidden_flatten_state,
                    log_var_hidden_flatten_state,
                    last_hidden_flatten_state,
                    patched_x,
                    loc,
                    scale,
                ]
            )

        return TSBeastBaseModelOutput(
            last_hidden_state=encoder_output.last_hidden_state,
            hidden_states=hidden_states,
            mu_hidden_flatten_state=mu_hidden_flatten_state,
            log_var_hidden_flatten_state=log_var_hidden_flatten_state,
            last_hidden_flatten_state=last_hidden_flatten_state,
            patch_input=patched_x,
            loc=loc,
            scale=scale,
        )


@dataclass
class TSBeastBaseForReconstructionOutput(ModelOutput):
    """
    Output type of [`TSBeastBaseForReconstructionOutput`].

    Args:
        reconstruction_outputs (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Reconstruction output from the reconstruction head.
        backbone_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Backbone embeddings before passing through the decoder
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Decoder embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        loss (*optional*, returned when `y` is provided, `torch.FloatTensor` of shape `()`):
            Total loss.
        loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean
        scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input std dev

    """

    loss: Optional[torch.FloatTensor] = None
    reconstruction_outputs: torch.FloatTensor = None
    backbone_hidden_state: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    loc: torch.FloatTensor = None
    scale: torch.FloatTensor = None


@dataclass
class TSBeastBaseDecoderWithReconstructionHeadOutput(ModelOutput):
    """
    Output type of [`TSBeastBaseDecoderWithReconstructionHeadOutput`].

    Args:
        reconstruction_outputs (`torch.FloatTensor` of shape `(batch_size, context_length, num_input_channels)`):
            Reconstruction output from the reconstruction head.
        decoder_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_input_channels, num_patches, d_model)`):
            Decoder embeddings before passing through the head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.

    """

    reconstruction_outputs: torch.FloatTensor = None
    decoder_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TSBeastBaseForReconstruction(TSBeastBasePreTrainedModel):
    r"""
    `TSBeastBase` for forecasting application.

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.

    Returns:
        `None`.
    """

    def __init__(self, config: TSBeastBaseConfig):
        super().__init__(config)
        config.check_and_init_preprocessing()

        self.loss = config.loss

        self.use_return_dict = config.use_return_dict

        self.backbone = TSBeastBaseModel(config)

        self.decoder_with_head = TSBeastBaseDecoderWithReconstructionHead(config)

        self.variational = config.variational

        self.kl_loss_weight = config.kl_loss_weight

        if config.decoder_d_model != config.d_model:
            raise Exception("decoder_d_model should be same as d_model for reconstruction tasks")

        if config.decoder_d_model_layerwise[-1] != config.d_model:
            raise ValueError("decoder_d_model_layerwise[-1] should be same as config.d_model")

        if config.decoder_num_patches_layerwise[-1] != config.num_patches:
            raise ValueError("decoder_num_patches_layerwise[-1] should be same as config.num_patches")

        if config.decoder_num_channels_layerwise[-1] != config.num_input_channels:
            raise ValueError("decoder_num_channels_layerwise[-1] should be same as config.num_input_channels")

        # Initialize weights and apply final processing
        if config.post_init:
            self.post_init()

    @add_start_docstrings_to_model_forward(TSBEASTBASE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TSBeastBaseForReconstructionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_loss: bool = True,
        return_dict: Optional[bool] = None,
    ) -> TSBeastBaseForReconstructionOutput:
        r"""
        observed_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_input_channels)`, *optional*):
            Boolean mask to indicate which `past_values` were observed and which were missing. Mask values selected
            in `[0, 1]`:
                - 1 for values that are **observed**,
                - 0 for values that are **missing** (i.e. NaNs that were replaced by zeros).
        return_loss (`bool`,  *optional*):
            Whether to return the loss in the `forward` call.

        Returns:

        """

        if self.loss == "mse":
            loss = nn.MSELoss(reduction="mean")
        else:
            raise ValueError("Invalid loss function: Allowed values: mse")

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # past_values: tensor [batch_size x context_length x num_input_channels]
        model_output = self.backbone(
            past_values,
            observed_mask=observed_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # model_output: [batch_size x nvars x num_patch x d_model]

        if isinstance(model_output, tuple):
            model_output = TSBeastBaseModelOutput(*model_output)

        decoder_input = model_output.last_hidden_flatten_state
        hidden_states = model_output.hidden_states

        loc = model_output.loc
        scale = model_output.scale

        decoder_with_head_output = self.decoder_with_head(
            decoder_input=decoder_input,
            loc=loc,
            scale=scale,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if isinstance(decoder_with_head_output, tuple):
            decoder_with_head_output = TSBeastBaseDecoderWithReconstructionHeadOutput(*decoder_with_head_output)

        if output_hidden_states:
            hidden_states.extend(decoder_with_head_output.hidden_states)

        loss_val = None

        if return_loss is True:
            loss_val = loss(decoder_with_head_output.reconstruction_outputs, past_values)

        if self.variational:
            log_var = model_output.log_var_hidden_flatten_state
            mu = model_output.mu_hidden_flatten_state
            kl_loss = -0.5 * torch.mean(1 + log_var - mu**2 - torch.exp(log_var))
            loss_val = loss_val + self.kl_loss_weight * kl_loss

        if not return_dict:
            return tuple(
                v
                for v in [
                    loss_val,
                    decoder_with_head_output.reconstruction_outputs,
                    model_output.last_hidden_state,
                    decoder_with_head_output.decoder_hidden_state,
                    hidden_states,
                    loc,
                    scale,
                ]
            )

        return TSBeastBaseForReconstructionOutput(
            loss=loss_val,
            reconstruction_outputs=decoder_with_head_output.reconstruction_outputs,  # tensor [batch_size x context_length x num_input_channels]
            backbone_hidden_state=model_output.last_hidden_flatten_state,  # x: [batch_size x nvars x num_patch x d_model]
            decoder_hidden_state=decoder_with_head_output.decoder_hidden_state,  # x: [batch_size x nvars x num_patch x decoder_d_model]
            hidden_states=hidden_states,
            loc=loc,
            scale=scale,
        )


class TSBeastBaseDecoderWithReconstructionHead(TSBeastBasePreTrainedModel):
    """
    Decoder + Head

    Args:
        config (`TSBeastBaseConfig`, *required*):
            Configuration.
    """

    def __init__(self, config: TSBeastBaseConfig):
        if not config.init_processing:
            config.check_and_init_preprocessing()
        super().__init__(config)
        self.use_return_dict = config.use_return_dict

        self.decoder = TSBeastBaseDecoder(config)
        self.head = TSBeastBaseForReconstructionHead(config)

        self.reshape_dims = (
            -1,
            config.num_channels_layerwise[-1],
            config.num_patches_layerwise[-1],
            config.d_model_layerwise[-1],
        )

        # # Initialize weights and apply final processing
        # if config.post_init:
        #     self.post_init()

    @replace_return_docstrings(
        output_type=TSBeastBaseDecoderWithReconstructionHeadOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        decoder_input: torch.Tensor,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
    ) -> TSBeastBaseDecoderWithReconstructionHeadOutput:
        r"""
        Args:
            decoder_input `torch.Tensor` of shape `(batch_size x emb_size)`): The input tensor from backbone.

            loc (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
            Input mean

            scale (`torch.FloatTensor`, *optional* of shape `(batch_size, 1, num_input_channels)`):
                Input std dev

            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.

            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        """

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        decoder_input = decoder_input.reshape(self.reshape_dims)

        decoder_output, decoder_hidden_states = self.decoder(decoder_input, output_hidden_states)

        reconstruction_outputs = self.head(decoder_output)

        if loc != None:
            reconstruction_outputs = reconstruction_outputs * scale + loc

        if output_hidden_states:
            decoder_hidden_states.append(reconstruction_outputs)

        if not return_dict:
            return tuple(
                v
                for v in [
                    reconstruction_outputs,
                    decoder_output,
                    decoder_hidden_states,
                ]
            )

        return TSBeastBaseDecoderWithReconstructionHeadOutput(
            reconstruction_outputs=reconstruction_outputs,
            decoder_hidden_state=decoder_output,
            hidden_states=decoder_hidden_states,
        )

