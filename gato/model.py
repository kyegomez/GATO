
import torch
import torch.nn as nn
import torch.nn.functional as F
from gato.attention import Attend

# embeddings
def _randomized_positions(from_v, to_v):
    pos = torch.rand_like(from_v) * (to_v - from_v)
    return pos.int()


def _rounded_mean_positions(from_v, to_v):
    pos = (from_v + to_v).float() / 2
    return pos.round()


# tokenizer


def mu_law_encode(x, mu=100, m=256):
    numerator = torch.log(x.abs(), *mu + 1.0)
    denominator = torch.log(m * mu + 1.0)
    return (numerator / denominator) * x.sign()


def tokenize_continous_value(x, mu=100, m=256, bins=1024, shift=None):
    # appenddix B agent data tokenization
    # finally they are discretized using bins of uniform width on the
    # domain[-1, 1]
    x = mu_law_encode(x, mu, m)

    # we use 1024 bins and shift the resulting integers
    # so they are not overlapping with the ones used for text tokens
    c = (c + 1) * (bins / 2)  # noqa: F821
    c = c.int()
    if shift is not None:
        c += shift
    return c


# EMBEDDINGS


class PatchPositionEncoding(nn.Module):
    def __init__(self, embedding_dim, discretize_depth, img_patch_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.discretize_depth = discretize_depth
        self.img_patch_size = img_patch_size

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

        return (
            input_ids
            + self.row_embedding(row_pos.long())
            + self.col_embedding(col_pos.long())
        )


class ResidualUnit(nn.Module):
    def __init__(self, num_groups: int, filters: int):
        super().__init__()
        self.num_groups = num_groups
        self.filters = filters

        self.conv1 = nn.Conv2d(
            in_channels=filters,
            out_channels=filters // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=filters // 2,
            out_channels=filters,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv_proj = nn.Conv2d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=1,
            stride=2,
            padding=0,
        )

        self.gn1 = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters)
        self.gn2 = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters // 2)
        self.gn_proj = nn.GroupNorm(num_groups=self.num_groups, num_channels=filters)

    def forward(self, x):
        residual = self.conv_prok(self.gn_proj(x))

        x = F.gelu(self.gn1(x))
        x = self.conv1(x)

        x = F.gelu(self.gn2(x))
        x = self.conv2(x)

        return x + residual


class ResidualEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,
        num_group_norm_groups,
        layer_width,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_group_norm_groups = num_group_norm_groups
        self.layer_width = layer_width

        self.root_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_dim,
                out_channels=96,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.GroupNorm(num_channels=96, num_groups=self.num_group_norm_groups),
            nn.GELU(),
        )

        self.residual_units = nn.ModuleList(
            [
                ResidualUnit(
                    num_groups=self.num_group_norm_groups, filters=96 * 2 ** (i + 1)
                )
                for i in range(3)
            ]
        )

        if self.input_dim != self.layer_width:
            self.conv_proj = nn.Conv2d(
                in_channels=96 * 2**3,
                out_channels=self.layer_width,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, images):
        x = self.root_conv(images)

        for unit in self.residual_units:
            x = unit(x)

        if self.input_dim != self.layer_width:
            x = self.conv_proj(x)

        return x


class LocalPositionEncoding(nn.Module):
    def __init__(
        self,
        token_sequence_length,
        layer_width,
        trainable=True,
        name=None,
        *args,
        **kwargs,
    ):
        """
        Appendix C.3. Position Encodings > Local Observation Position Encodings
        """
        super(LocalPositionEncoding, self).__init__()
        self.token_sequence_length = token_sequence_length
        self.layer_width = layer_width
        self.trainable = trainable
        self.name = name

        self.embedding = nn.Embedding(self.token_sequence_length, self.layer_width)

    def forward(self, inputs):
        obs_pos, obs_mask = inputs
        embed = self.embedding(obs_pos)

        ones = torch.ones((embed.shape[0], 1, self.layer_width)).to(embed.device)
        obs_mask = obs_mask.float().transpose(-1, -2).matmul(ones)
        return embed * obs_mask


class DiscreteEmbedding(nn.Module):
    def __init__(
        self,
        embedding_input_size,
        layer_width,
    ):
        super(DiscreteEmbedding, self).__init__()
        self.embedding_input_size = embedding_input_size
        self.layer_width = layer_width

        self.embedding = nn.Embedding(self.embedding_input_size, self.layer_width)

    def forward(self, inputs):
        return self.embedding(inputs)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_patch_size,
        input_dim,
        num_group_norm_groups,
        layer_width,
        discretize_depth,
    ):
        super(PatchEmbedding, self).__init__()
        self.img_patch_size = img_patch_size
        self.input_dim = input_dim
        self.num_group_norm_groups = num_group_norm_groups
        self.discretize_depth = discretize_depth
        self.layer_width = layer_width

        self.residual_embedding = ResidualEmbedding(
            self.input_dim, self.num_group_norm_groups, self.layer_width
        )
        self.pos_encoding = PatchPositionEncoding(
            self.input_dim, self.discretize_depth, self.img_patch_size
        )

    def forward(self, inputs):
        input_ids, (row_pos, col_pos) = inputs
        patch_size = self.img_patch_size
        depth = self.input_dim // (patch_size * patch_size)

        x = input_ids.view(-1, input_ids.size(1), patch_size, patch_size, depth)
        x = x.permute(0, 1, 4, 2, 3).contiguous()

        x = self.residual_embedding(x)
        x = self.pos_encoding((x, (row_pos, col_pos)))
        return x


class ContinousValueTokenizer(nn.Module):
    def __init__(self, vocab_size, mu=100, m=256, bins=1024):
        super(ContinousValueTokenizer, self).__init__()
        self.vocab_size = vocab_size
        self.mu = mu
        self.m = m
        self.bins = bins

    def forward(self, inputs):
        return tokenize_continous_value(
            inputs, self.mu, self.m, self.bins, shift=self.vocab_size
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dropout_rate,
        layer_width,
        feedforward_hidden_size,
    ):
        super(TransformerBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.layer_width = layer_width
        self.feedforward_hidden_size = feedforward_hidden_size

        self.attention = Attend(
        )

        # may be unnecessary
        self.dropout = nn.Dropout(self.dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=self.layer_width, out_features=self.feedforward_hidden_size
            ),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(
                in_features=self.feedforward_hidden_size, out_features=self.layer_width
            ),
            nn.Dropout(self.dropout_rate),
        )

        self.layer_norm1 = nn.LayerNorm(normalized_shape=self.layer_width, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=self.layer_width, eps=1e-6)

    def forward(self, inputs):
        x_norm1 = self.layer_norm1(inputs)
        x_attention, _ = self.attention(x_norm1, x_norm1, x_norm1)
        x_dropout = self.dropout(x_attention)

        x_residual = x_dropout + inputs
        x_norm2 = self.layer_norm2(x_residual)
        x_ff = self.feed_forward(x_norm2)

        x_residual2 = x_ff + x_residual

        return x_residual2


class Transformer(nn.Module):
    def __init__(
        self,
        num_transformer_blocks,
        dropout_rate,
        layer_width,
        feedforward_hidden_size,
    ):
        super(Transformer, self).__init__()
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.layer_width = layer_width
        self.feedforward_hidden_size = feedforward_hidden_size

        self.encoders = nn.ModuleList(
            [
                TransformerBlock(
                    self.dropout_rate, self.layer_width, self.feedforward_hidden_size
                )
                for _ in range(self.num_transformer_blocks)
            ]
        )

    def forward(self, inputs):
        x = inputs
        for encoder in self.encoders:
            x = encoder(x)

        return x


class Gato(nn.Module):
    def __init__(
        self,
        img_patch_size: int = 16,
        input_dim: int = 768,
        token_sequence_length: int = 1024,
        vocabulary_size: int = 32000,
        actions_size: int = 1024,
        continuous_values_size: int = 1024,
        num_transformer_blocks: int = 8,
        num_attention_heads: int = 24,
        layer_width: int = 768,
        feedforward_hidden_size: int = 3072,
        key_value_size: int = 32,
        dropout_rate: float = 0.1,
        num_group_norm_groups: int = 32,
        discretize_depth: int = 128,
        local_position_encoding_size: int = 512,
        max_seq_len: int = 8192,
    ):
        super(Gato, self).__init__()

        self.img_patch_size = img_patch_size
        self.input_dim = input_dim
        self.token_sequence_length = token_sequence_length

        self.vocabulary_size = vocabulary_size
        self.actions_size = actions_size
        self.continuous_values_size = continuous_values_size

        self.num_transformer_blocks = num_transformer_blocks
        self.num_attention_heads = num_attention_heads
        self.layer_width = layer_width

        self.feedforward_hidden_size = feedforward_hidden_size
        self.key_value_size = key_value_size
        self.dropout_rate = dropout_rate

        self.num_group_norm_groups = num_group_norm_groups
        self.discretize_depth = discretize_depth
        self.local_position_encoding_size = local_position_encoding_size

        self.max_seq_len = max_seq_len

        self.image_embedding = PatchEmbedding(
            self.img_patch_size,
            self.input_dim,
            self.num_group_norm_groups,
            self.layer_width,
            self.discretize_depth,
        )

        self.discrete_embedding = DiscreteEmbedding(self.input_dim, self.layer_width)

        self.continuous_encoding = ContinousValueTokenizer(self.vocabulary_size)

        self.transformer = Transformer(
            self.num_transformer_blocks,
            self.dropout_rate,
            self.layer_width,
            self.feedforward_hidden_size,
        )

        self.local_pos_encoding = LocalPositionEncoding(
            self.token_sequence_length, self.layer_width
        )

    def forward(self, inputs):
        input_ids, (encoding, row_pos, col_pos), (obs_pos, obs_mask) = inputs
        encoding = F.one_hot(encoding, num_classes=3).float()

        ones = torch.ones((input_ids.size(0), 1, self.layer_width))

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
