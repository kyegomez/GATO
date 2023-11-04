import torch
from gato import Gato

# Full usage example for Gato class

# Import necessary libraries
import torch

# Assume we have some synthetic data for simplicity:
batch_size = 4
num_patches = 10
patch_size = 16
img_channels = 3
sequence_length = 1024
vocabulary_size = 32000
num_continuous_features = 10
num_actions = 1024

# Create synthetic data
image_patches = torch.rand(batch_size, num_patches, patch_size*patch_size*img_channels)
text_tokens = torch.randint(low=0, high=vocabulary_size, size=(batch_size, sequence_length))
continuous_values = torch.rand(batch_size, num_continuous_features) * 2 - 1  # Continuous values in range [-1, 1]
robot_actions = torch.randint(low=0, high=num_actions, size=(batch_size, num_actions))

# Initialize Gato
gato = Gato(
    img_patch_size=patch_size,
    input_dim=patch_size*patch_size*img_channels,
    token_sequence_length=sequence_length,
    vocabulary_size=vocabulary_size,
    actions_size=num_actions,
    continuous_values_size=num_continuous_features,
    num_transformer_blocks=8,
    num_attention_heads=24,
    layer_width=768,
    feedforward_hidden_size=3072,
    key_value_size=32,
    dropout_rate=0.1,
    num_group_norm_groups=32,
    discretize_depth=128,
    local_position_encoding_size=512,
    max_seq_len=8192,
)

# Forward pass through Gato with synthetic data
transformed_output = gato(image_patches, text_tokens, continuous_values, robot_actions)

# Print the output shape
print("Transformed output shape:", transformed_output.shape)
