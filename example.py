import torch
from torch import nn
from torch.nn import functional as F

from gato import Gato

# Assuming the necessary modules (PatchEmbedding, DiscreteEmbedding, ContinousValueTokenizer, Transformer, LocalPositionEncoding) are defined

# Initialize the model
model = Gato()

batch_size = 2
seq_len = 1024
layer_width = 768
patch_size = 16
depth = 3
# Create random inputs
image_height = 16
image_width = 16
num_channels = 3

input_ids = torch.randint(0, 32000, (batch_size, num_channels, image_height, image_width)).float()
encoding = torch.randint(0, 3, (batch_size, seq_len))
row_pos = torch.randint(0, 16, (batch_size, seq_len))
col_pos = torch.randint(0, 16, (batch_size, seq_len))
obs_pos = torch.randint(0, 8192, (batch_size, seq_len))
obs_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

inputs = (input_ids, (encoding, row_pos, col_pos), (obs_pos, obs_mask))

# Forward pass
hidden_states = model(inputs)

print(hidden_states.shape)  # Expected: [batch_size, seq_len, layer_width]