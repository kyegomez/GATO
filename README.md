[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

<h1 align="center">Gato: A Generalist Agent</h1>

[[Deepmind Publication]](https://www.deepmind.com/publications/a-generalist-agent)
[[arXiv Paper]](https://arxiv.org/pdf/2205.06175.pdf)

* Please Help with this open source implementation in the Agora discord, ![Discord](https://img.shields.io/discord/999382051935506503)


### Installation

```bash
$ pip install gato-torch
```

```python
import torch
from gato import Gato

#create model instance
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


#fake inputs for Gato
input_dim = config.input_dim
input_ids = torch.cat([
    torch.rand((1, 1, input_dim)) for _ in range(20)] + # 20 image patches
    [torch.full((1, 1, input_dim), 0.25), #continous value]
     torch.full((1, 1, input_dim), 624.0)] + #discrete (actions, texts)
     [torch.rand((1, 1, input_dim)) for _ in range(20)] + #20 image patches
     [torch.full((1, 1, input_dim), 0.12), #continous value
      torch.full((1, 1, input_dim), 295.0)], #discrete( actions, text)
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
```



### Dataset and Model Architecture
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/5837620/215323793-7f7bcfdb-d8be-40d3-8e58-a053511f95d5.png">
  <img alt="gato dataset and model architecture" src="https://user-images.githubusercontent.com/5837620/215323795-3a433516-f5ca-4272-9999-3df87ae521ba.png">
</picture>

## Paper Reviews

### Full Episode Sequence

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://user-images.githubusercontent.com/5837620/175756389-31d183c9-054e-4829-93a6-df79781ca212.png">
    <img alt="gato dataset architecture" src="https://user-images.githubusercontent.com/5837620/175756409-75605dbc-7756-4509-ba93-c0ad08eea309.png">
</picture>

### Architecture Variants

> Appendix C.1. Transformer Hyperparameters

In the paper, Deepmind tested Gato with 3 architecture variants, `1.18B`, `364M`, and `79M`.<br>
I have named them as `large()`, `baseline()` and `small()` respectively in `GatoConfig`.

| Hyperparameters          | Large(1.18B) | Baseline(364M) | Small(79M) |
|--------------------------|--------------|----------------|------------|
| Transformer blocks       | 24           | 12             | 8          |
| Attention heads          | 16           | 12             | 24         |
| Layer width              | 2048         | 1536           | 768        |
| Feedforward hidden size  | 8192         | 6144           | 3072       |
| Key/value size           | 128          | 128            | 32         |


### Residual Embedding

> Appendix C.2. Embedding Function

There are no mentions that how many residual networks must be stacked for token embeddings.<br>
Therefore, I remain configurable in `GatoConfig`.

Whatever how many residual layers are existing, full-preactivation is a key.

The blocks are consisted of:
- Version 2 ResNet architecture (based on ResNet50V2)
- GroupNorm (instead of LayerNorm)
- GeLU (instead of ReLU)

### Position Encodings

> Appendix C.3. Position Encodings

#### Patch Position Encodings

Like [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer) by Google, Gato takes the input images as raster-ordered 16x16 patches.<br>
Unlike the Vision Transformer model, however, Gato divides its patch encoding strategy into 2 phases, training and evaluation.

For high-performance computation in TensorFlow, I have used the following expressions.

$C$ and $R$ mean column and row-wise, and $F$ and $T$ mean `from` and `to` respectively.

$$
\begin{align}
  v^R_F &= \begin{bmatrix}
    0 & 32 & 64 & 96
  \end{bmatrix} \\
  v^R_T &= \begin{bmatrix}
    32 & 64 & 96 & 128
  \end{bmatrix} \\
  v^C_F &= \begin{bmatrix}
    0 & 26 & 51 & 77 & 102
  \end{bmatrix} \\
  v^C_T &= \begin{bmatrix}
    26 & 51 & 77 & 102 & 128
  \end{bmatrix} \\
  \\
  P_R &= \begin{cases}
    \mathsf{if} \ \mathsf{training} & v^R_F + \mathsf{uniform}(v^R_T - v^R_F) \\
    \mathsf{otherwise} & \mathsf{round}(\frac{v^R_F + v^R_T}{2})
  \end{cases} \\
  P_C &= \begin{cases}
    \mathsf{if} \ \mathsf{training} & v^C_F + \mathsf{uniform}(v^C_T - v^C_F) \\
    \mathsf{otherwise} & \mathsf{round}(\frac{v^C_F + v^C_T}{2})
  \end{cases} \\
  \\
  E^R_P &= P_R \cdot 1^{\mathsf{T}}_C \\
  E^C_P &= 1^{\mathsf{T}}_R \cdot P_C \\
  \\
  \therefore E &= E_I + E^R_P + E^C_P
\end{align}
$$

#### Local Observation Position Encodings

In the definition of Appendix B., text tokens, image patch tokens, and discrete & continuous values are observation tokens<br>
When Gato receives those values, they must be encoded with their own (local) time steps.


## Contributing
[We welcome all contributions, please either submit a pull request or submit issues in the Agora discord](https://discord.gg/qUtxnK2NMf)

## License
Licensed under the [MIT license](/LICENSE).

# Roadmap:

* Get functional prototype

* Integrate ALIBI, multi query, qk norm and other SOTA stuff

* integrate action tokens

