from torch import nn
from src.models.encoders import get_encoder


class Student(nn.Module):
    def __init__(
        self,
        encoder_name,
        num_classes,
        hidden_dim,
        pretrained=True,
        freeze_encoder=True,
    ):
        super().__init__()

        self.encoder, embedding_dim = get_encoder(
            encoder_name,
            pretrained,
        )

        for p in self.encoder.parameters():
            p.requires_grad = True

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.mlp(z)

    def mlp_forward(self, z):
        return self.mlp(z)
