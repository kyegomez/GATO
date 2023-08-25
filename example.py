import torch
from gato import Gato

# Create model instance
# Create an instance of Gato
gato = Gato(input_dim=768,
            img_patch_size=16,
            token_sequence_length=1024,
            vocabulary_size=32000,
            actions_size=1024,
            continuous_values_size=1024,
            num_transformer_blocks=8,
            num_attention_heads=24,
            layer_width=768,
            feedforward_hidden_size=3072,
            key_value_size=32,
            dropout_rate=0.1,
            num_group_norm_groups=32,
            discretize_depth=128,
            local_position_encoding_size=512,
            max_seq_len=8192)

# Fake inputs for Gato
input_dim = 768
input_ids = torch.cat([
    torch.rand((1, 1, input_dim)) for _ in range(20)] +  # 20 image patches
    [torch.full((1, 1, input_dim), 0.25),  # continuous value
    torch.full((1, 1, input_dim), 624.0)] +  # discrete (actions, texts)
    [torch.rand((1, 1, input_dim)) for _ in range(20)] +  # 20 image patches
    [torch.full((1, 1, input_dim), 0.12),  # continuous value
    torch.full((1, 1, input_dim), 295.0)],  # discrete (actions, texts)
    dim=1)
encoding = torch.tensor([
    [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2]
])
row_pos = (
    torch.tensor([[0.00, 0.25, 0.50, 0.75, 0, 0, 0.00, 0.25, 0.50, 0.75, 0, 0]]),  # pos_from
    torch.tensor([[0.25, 0.50, 0.75, 1.00, 0, 0, 0.25, 0.50, 0.75, 1.00, 0, 0]])  # pos_to
)
col_pos = (
    torch.tensor([[0.00, 0.00, 0.00, 0.80, 0, 0, 0.00, 0.00, 0.00, 0.80, 0, 0]]),  # pos_from
    torch.tensor([[0.20, 0.20, 0.20, 1.00, 0, 0, 0.20, 0.20, 0.20, 1.00, 0, 0]])  # pos_to
)
obs = (
    torch.tensor([[ 0,  1,  2, 19, 20, 21,  0,  1,  2, 19, 20, 21]]),  # obs token
    torch.tensor([[ 1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0]])  # obs token masking (for action tokens)
)
hidden_states = gato((input_ids, (encoding, row_pos, col_pos), obs))
