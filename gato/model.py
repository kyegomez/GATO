import copy
from collections import namedtuple
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from packaging import version
from torch import Tensor, einsum

# constants

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# helpers

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class



@dataclass
class Intermediates:
    """
    Dataclass to store intermediate tensors during attention computation.

    Args:
        qk_similarities (torch.Tensor): Tensor storing the similarities between query and key.
        pre_softmax_attn (torch.Tensor): Tensor storing the attention weights before softmax.
        post_softmax_attn (torch.Tensor): Tensor storing the attention weights after softmax.

    Methods:
        to_tuple(): Convert the Intermediates object to a tuple.

    """
    qk_similarities: Tensor = None
    pre_softmax_attn: Tensor = None
    post_softmax_attn: Tensor = None

    def to_tuple(self):
        """
        Convert the Intermediates object to a tuple.

        Returns:
            tuple: Tuple representation of the Intermediates object.
        """
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)


class FlashAttention(nn.Module):
    def __init__(
        self,
        causal: bool = False,
        dropout: float  = 0.,
        flash: bool = True
    ):
        """
        FlashAttention module that performs attention computation.

        Args:
            causal (bool): Whether to apply causal masking (default: False).
            dropout (float): Dropout probability (default: 0.).
            flash (bool): Whether to use flash attention (default: True).

        """
        super().__init__()

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = EfficientAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = EfficientAttentionConfig(False, True, True)

    def get_mask(self, i, j, device):
        """
        Generate a mask for attention computation.

        Args:
            i (int): Length of the query sequence.
            j (int): Length of the key sequence.
            device (torch.device): Device to place the mask tensor.

        Returns:
            torch.Tensor: Mask tensor of shape (i, j).

        """
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)


    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None
    ):
        
        """
        Perform flash attention computation.

        Args:
            q (torch.Tensor): Query tensor of shape (batch, heads, q_len, dim).
            k (torch.Tensor): Key tensor of shape (batch, heads, k_len, dim).
            v (torch.Tensor): Value tensor of shape (batch, heads, v_len, dim).
            mask (torch.Tensor): Mask tensor of shape (batch, heads, q_len, k_len) (default: None).
            attn_bias (torch.Tensor): Attention bias tensor of shape (batch, heads, q_len, k_len) (default: None).

        Returns:
            torch.Tensor: Output tensor of shape (batch, heads, q_len, dim).

        """
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention
        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

            # manually handle causal mask, if another mask was given

            if causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                mask = mask & ~causal_mask
                causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

            return out

    def forward(self, q, k, v, mask = None, attn_bias = None):
        """
        Perform attention computation.

        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension

        Args:
            q (torch.Tensor): Query tensor of shape (batch, heads, q_len, dim).
            k (torch.Tensor): Key tensor of shape (batch, heads, k_len, dim).
            v (torch.Tensor): Value tensor of shape (batch, heads, v_len, dim).
            mask (torch.Tensor): Mask tensor of shape (batch, heads, q_len, k_len) (default: None).
            attn_bias (torch.Tensor): Attention bias tensor of shape (batch, heads, q_len, k_len) (default: None).

        Returns:
            torch.Tensor: Output tensor of shape (batch, heads, q_len, dim).

        """


        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # attention bias

        if exists(attn_bias):
            sim = sim + attn_bias

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
    

# embeddings
def _randomized_positions(from_v, to_v):
    pos = torch.rand_like(from_v) * (to_v - from_v)
    return pos.int()


def _rounded_mean_positions(from_v, to_v):
    pos = (from_v + to_v).float() / 2
    return pos.round()


# tokenizer

def mu_law_encode(x, mu=100, m=256):
    numerator = torch.log(x.abs(), * mu + 1.0)
    denominator = torch.log(m * mu + 1.0)
    return (numerator / denominator) * x.sign()


def tokenize_continous_value(x, mu=100, m=256, bins=1024, shift=None):
    #appenddix B agent data tokenization
    #finally they are discretized using bins of uniform width on the domain[-1, 1]
    x = mu_law_encode(x, mu, m)

    #we use 1024 bins and shift the resulting integers
    #so they are not overlapping with the ones used for text tokens
    c = (x + 1) * (bins / 2)  # noqa: F821
    c = c.int()
    if shift is not None:
        c += shift
    return c

# config

class GatoConfig:
    @staticmethod
    def large():
        return GatoConfig(num_transformer_blocks=24,
                          num_attention_heads=16,
                          layer_width=2048,
                          feedforward_hidden_size=8192,
                          key_value_size=128)

    @staticmethod
    def baseline():
        return GatoConfig(num_transformer_blocks=12,
                          num_attention_heads=12,
                          layer_width=1536,
                          feedforward_hidden_size=6144,
                          key_value_size=128)

    @staticmethod
    def small():
        return GatoConfig(num_transformer_blocks=8,
                          num_attention_heads=24,
                          layer_width=768,
                          feedforward_hidden_size=3072,
                          key_value_size=32)

    def __init__(self, **kwargs):
        self.input_dim = kwargs.pop('input_dim', 768)
        self.img_patch_size = kwargs.pop('img_patch_size', 16)

        # Section 2.3. Training
        self.token_sequence_length = kwargs.pop('token_sequence_length', 1024)

        # Section 2.1. Tokenization
        # Text - SentencePiece
        self.vocabulary_size = kwargs.pop('vocabulary_size', 32000)
        # Discrete values
        self.actions_size = kwargs.pop('actions_size', 1024)
        # Continuous values
        self.continuous_values_size = kwargs.pop('continuous_values_size', 1024)

        # Appendix C.1. Transformer Hyperparameters
        self.num_transformer_blocks = kwargs.pop('num_transformer_blocks', 8)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 24)
        self.layer_width = kwargs.pop('layer_width', 768)
        self.feedforward_hidden_size = kwargs.pop('feedforward_hidden_size', 3072)
        self.key_value_size = kwargs.pop('key_value_size', 32)

        # Appendix E. Regularization
        self.dropout_rate = kwargs.pop('dropout_rate', 0.1)

        # Appendix C.2. Embedding Function
        self.num_group_norm_groups = kwargs.pop('num_group_norm_groups', 32)

        # Appendix C.3. Position Encodings > Patch Position Encodings
        self.discretize_depth = kwargs.pop('discretize_depth', 128)
        # Appendix C.3. Position Encodings > Local Observation Position Encodings
        self.local_position_encoding_size = kwargs.pop('local_position_encoding_size', 512)

        self.max_seq_len = kwargs.pop('max_seq_len', 8192)

    @property
    def embedding_input_size(self):
        return self.vocabulary_size + self.continuous_values_size + self.actions_size + 1

    @property
    def output_target_size(self):
        return self.vocabulary_size + self.actions_size

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GatoConfig":
        config = cls(**config_dict)
        return config
    

#EMBEDDINGS




class PatchPositionEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_dim = config.layer_width
        self.discretize_depth = config.discretize_depth
        self.patch_size = config.img_patch_size

        self.row_embedding = nn.Embedding(self.discretize_depth, self.embedding_dim)
        self.col_embedding = nn.Embedding(self.discretize_depth, self.embedding_dim)

    def _discretize(self, pos):
        return (pos * self.discretize_depth).round()

    def _discretize_interval(self, interval):
        pos_from, pos_to = interval
        return self._discretize(pos_from), self._discretize(pos_to)

    def forward(self, input_ids, pos):
        row_pos, col_pos = pos

        row_pos_from, row_pos_to = self._discretize_interval(row_pos)
        col_pos_from, col_pos_to = self._discretize_interval(col_pos)

        if self.training:
            row_pos = row_pos_from + _randomized_positions(row_pos_from, row_pos_to)
            col_pos = col_pos_from + _randomized_positions(col_pos_from, col_pos_to)
        else:
            row_pos = _rounded_mean_positions(row_pos_from, row_pos_to)
            col_pos = _rounded_mean_positions(col_pos_from, col_pos_to)

        return input_ids + self.row_embedding(row_pos.long()) + self.col_embedding(col_pos.long())

    
    def get_config(self):
        config = super(PatchPositionEncoding, self).get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config


class ResidualUnit(nn.Module):
    def __init__(self, num_groups: int, filters: int):
        super().__init__()
        self.num_groups = num_groups
        self.filters = filters
        self.conv1 = nn.Conv2d(in_channels=filters, out_channels=filters//2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filters//2, out_channels=filters, kernel_size=3, stride=2, padding=1)
        
        self.conv_proj = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=2, padding=0)
        self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters)
        self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters//2)
        self.gn_proj = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters)

    def forward(self, x):
        residual = self.conv_prok(self.gn_proj(x))

        x = F.gelu(self.gn1(x))
        x = self.conv1(x)

        x = F.gelu(self.gn2(x))
        x = self.conv2(x)

        return x + residual



class ResidualEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.root_conv = nn.Sequential(
            nn.Conv2d(in_channels=config.input_dim, out_channels=96, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_channels=96, num_groups=config.num_group_norm_groups),
            nn.GELU()
        )

        self.residual_units = nn.ModuleList([ResidualUnit(num_groups=config.num_group_norm_groups,
                                                          filters=96 * 2 ** (i + 1))
                                                          for i in range(3)])
        
        if config.input_dim != config.layer_width:
            self.conv_proj = nn.Conv2d(in_channels=96 * 2 ** 3, out_channels=config.layer_width, kernel_size=1, stride=1, padding=0)
    

    def forward(self, images):
        x = self.root_conv(images)

        for unit in self.residual_units:
            x = unit(x)

        if self.config.input_dim != self.config.layer_width:
            x = self.conv_proj(x)

        return x
        

    def get_config(self):
        config = super(ResidualEmbedding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config




class LocalPositionEncoding(nn.Module):
    def __init__(self, 
                 config: Union[GatoConfig, Dict[str, Any]], 
                 trainable=True,
                 name=None, 
                 *args, **kwargs):
        """
        Appendix C.3. Position Encodings > Local Observation Position Encodings
        """
        super(LocalPositionEncoding, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.embedding = nn.Embedding(self.config.token_sequence_length, self.config.layer_width)

    def forward(self, inputs):
        obs_pos, obs_mask = inputs
        embed = self.embedding(obs_pos)

        ones = torch.ones((embed.shape[0], 1, self.config.layer_width)).to(embed.device)
        obs_mask = obs_mask.float().transpose(-1, -2).matmul(ones)
        return embed * obs_mask

    def get_config(self):
        config = super(LocalPositionEncoding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config


class DiscreteEmbedding(nn.Module):
    def __init__(self, config):
        super(DiscreteEmbedding, self).__init__()
        
        if isinstance(config, dict):
            config = GatoConfig(**config)

        self.config = config
        self.embedding = nn.Embedding(self.config.embedding_input_size, self.config.layer_width)

    def forward(self, inputs):
        return self.embedding(inputs)

    def get_config(self):
        config = super(DiscreteEmbedding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config
    

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super(PatchEmbedding, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.residual_embedding = ResidualEmbedding(config)
        self.pos_encoding = PatchPositionEncoding(config)
    
    def forward(self, inputs):
        input_ids, (row_pos, col_pos) = inputs
        patch_size = self.config.img_patch_size
        depth = self.config.input_dim // (patch_size * patch_size)

        x = input_ids.view(-1, input_ids.size(1), patch_size, patch_size, depth)
        x = self.residual_embedding(x)
        x = self.pos_encoding((x, (row_pos, col_pos)))
        return x

    def get_config(self):
        return super(PatchEmbedding, self).get_config()



class ContinousValueTokenizer(nn.Module):
    def __init__(self, config, mu=100, m=256, bins=1024):
        super(ContinousValueTokenizer, self).__init__()
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.mu = mu
        self.m = m
        self.bins = bins

    def forward(self, inputs):
        return tokenize_continous_value(inputs, self.mu, self.m, self.bins, shift=self.config.vocabulary_size)
    
    def get_config(self):
        return super(ContinousValueTokenizer, self).get_config()

    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.attention = FlashAttention(causal=True, dropout=0.1, flash=True)
        
        #may be unnecessary
        self.dropout = nn.Dropout(config.dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=config.layer_width, out_features=config.feedforward_hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(in_features=config.feedforward_hidden_size, out_features=config.layer_width),
            nn.Dropout(config.dropout_rate)
        )

        self.layer_norm1 = nn.LayerNorm(normalized_shape=config.layer_width, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=config.layer_width, eps=1e-6)


    def forward(self, inputs):
        x_norm1 = self.layer_norm1(inputs)
        x_attention, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x_dropout = self.dropout(x_attention)
        x_residual = x_dropout + inputs

        x_norm2 = self.layer_norm2(x_residual)
        x_ff = self.feed_forward(x_norm2)
        x_residual2 = x_ff + x_residual
        return x_residual2


    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config
    



class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.encoders = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_transformer_blocks)])

    def forward(self, inputs):
        x = inputs
        for encoder in self.encoders:
            x = encoder(x)
        
        return x

    def get_config(self):
        return super(Transformer, self).get_config()


class Gato(nn.Module):
    def __init__(self, config):
        super(Gato, self).__init__()

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.image_embedding = PatchEmbedding(config)
        self.discrete_embedding = DiscreteEmbedding(config)
        self.continuous_encoding = ContinousValueTokenizer(config)
        self.transformer = Transformer(config)
        self.local_pos_encoding = LocalPositionEncoding(config)

    def forward(self, inputs):
        input_ids, (encoding, row_pos, col_pos), (obs_pos, obs_mask) = inputs
        encoding = F.one_hot(encoding, num_classes=3).float()

        ones = torch.ones((input_ids.size(0), 1, self.config.layer_width))
        image_embed = self.image_embedding((input_ids, (row_pos, col_pos)))
        image_embed *= encoding[..., 0].unsqueeze(-1).matmul(ones)

        continuous_embed = self.continuous_encoding(input_ids[..., 0])
        continuous_embed = self.discrete_embedding(continuous_embed)
        continuous_embed *= encoding[..., 1].unsqueeze(-1).matmul(ones)

        discrete_embed = self.discrete_embedding(input_ids[..., 0])
        discrete_embed *= encoding[..., 2].unsqueeze(-1).matmul(ones)

        embed = image_embed + continuous_embed + discrete_embed
        embed += self.local_pos_encoding((obs_pos, obs_mask))

        hidden_states = self.transformer(embed)
        return hidden_states
    
    def get_config(self):
        return super(Gato, self).get_config()
    
