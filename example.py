import torch
from gato.model import Gato

# Create model instance

gato = Gato()

# Fake inputs for Gato
input_dim = gato.input_dim
input_ids = torch.cat([
  # ...
  # observation 1
  torch.rand((1, 1, input_dim)),  # image patch 0
  torch.rand((1, 1, input_dim)),  # image patch 1
  torch.rand((1, 1, input_dim)),  # image patch 2
  # ...
  torch.rand((1, 1, input_dim)),  # image patch 19
  torch.full((1, 1, input_dim), 0.25),  # continuous value
  torch.full((1, 1, input_dim), 624.0),  # discrete (actions, texts)

  # observation 2
  torch.rand((1, 1, input_dim)),  # image patch 0
  torch.rand((1, 1, input_dim)),  # image patch 1
  torch.rand((1, 1, input_dim)),  # image patch 2
  # ...
  torch.rand((1, 1, input_dim)),  # image patch 19
  torch.full((1, 1, input_dim), 0.12),  # continuous value
  torch.full((1, 1, input_dim), 295.0)  # discrete (actions, texts)
  # ...
], axis=1)
encoding = torch.tensor([
  # 0 - image patch embedding
  # 1 - continuous value embedding
  # 2 - discrete embedding (actions, texts)
  [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1, 2]
])
row_pos = (
  torch.tensor([[0.00, 0.25, 0.50, 0.75, 0, 0, 0.00, 0.25, 0.50, 0.75, 0, 0]]),  # pos_from
  torch.tensor([[0.25, 0.50, 0.75, 1.00, 0, 0, 0.25, 0.50, 0.75, 1.00, 0, 0]])   # pos_to
)
col_pos = (
  torch.tensor([[0.00, 0.00, 0.00, 0.80, 0, 0, 0.00, 0.00, 0.00, 0.80, 0, 0]]),  # pos_from
  torch.tensor([[0.20, 0.20, 0.20, 1.00, 0, 0, 0.20, 0.20, 0.20, 1.00, 0, 0]])   # pos_to
)
obs = (
  torch.tensor([[ 0,  1,  2, 19, 20, 21,  0,  1,  2, 19, 20, 21]]),  # obs token
  torch.tensor([[ 1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0]])   # obs token masking (for action tokens)
)
hidden_states = gato((input_ids, (encoding, row_pos, col_pos), obs))
