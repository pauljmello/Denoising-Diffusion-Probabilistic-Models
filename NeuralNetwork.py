import torch
from torch import nn


def sinusoidalEmbedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, steps, timeEmdedding=128):
        super(NeuralNetwork, self).__init__()

        self.dimensionality = hidden_dim

        self.activation = nn.ReLU()

        self.timeEmb = self.timeEmbed(timeEmdedding, 1)

        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = nn.Linear(hidden_dim, output_dim)

        self.time_embed = nn.Embedding(steps, timeEmdedding)
        self.time_embed.weight.data = sinusoidalEmbedding(steps, timeEmdedding)
        self.time_embed.requires_grad_(False)

    def timeEmbed(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

    def forward(self, x, t):
        n = len(x)
        t = self.time_embed(t)

        out = self.layer_1(x)
        out = self.activation(out)
        out = self.layer_2(out)
        out = self.activation(out)
        out = self.layer_3(out)
        out = self.activation(out)
        out = self.layer_4(out)
        out = self.activation(out)
        return out

    """
    def forward(self, x, t):
        n = len(x)
        t = self.time_embed(t)

        out = self.layer_1(x + self.timeEmb(t, self.dimensionality))
        out = self.activation(out)
        out = self.layer_2(out + self.timeEmb(t, self.dimensionality))
        out = self.activation(out)
        out = self.layer_3(out + self.timeEmb(t, self.dimensionality))
        out = self.activation(out)
        out = self.layer_4(out + self.timeEmb(t, self.dimensionality))
        out = self.activation(out)
        return out
    """