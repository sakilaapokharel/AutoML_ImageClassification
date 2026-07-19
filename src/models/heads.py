from torch import nn


class MLPHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims=(512,),
        dropout=0.2,
    ):
        super().__init__()

        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
