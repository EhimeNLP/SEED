import torch.nn as nn

class MLP_large(nn.Module):
    def __init__(self, emb_size=1024):
        super().__init__()
        self.meaning_emb_layer = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        meaning_emb = self.meaning_emb_layer(x)
        return meaning_emb
