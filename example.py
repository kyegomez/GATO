import torch
from gato.model import Gato

# Model hyperparameters
img_patch_size = 16  
input_dim = 768
token_sequence_length = 1024
layer_width = 768

# Create model
model = Gato(img_patch_size=img_patch_size, 
             input_dim=input_dim,
             token_sequence_length=token_sequence_length,
             layer_width=layer_width)

# Random input tensors  
batch_size = 4
seq_len = 32

input_ids = torch.randint(0, 255, (batch_size, seq_len, img_patch_size, img_patch_size, 3)).float() 
encoding = torch.randint(0, 3, (batch_size, seq_len))
row_pos = torch.rand(batch_size, seq_len)  
col_pos = torch.rand(batch_size, seq_len)
obs_pos = torch.randint(0, token_sequence_length, (batch_size, seq_len))
obs_mask = torch.ones(batch_size, seq_len)

inputs = (input_ids, (encoding, row_pos, col_pos), (obs_pos, obs_mask))

outputs = model(inputs)

print(outputs.shape)