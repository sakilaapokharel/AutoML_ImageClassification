from torch import nn


class Student(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_classes,
        hidden_dim=256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim // 2,
                num_classes,
            ),
        )

    def forward(self, x):
        return self.network(x)


# from torch import nn


# class Student(nn.Module):

#     def __init__(
#         self,
#         embedding_dim,
#         num_classes,
#         hidden_dim=512,
#         dropout=0.3,
#     ):
#         super().__init__()

#         self.network = nn.Sequential(

#             nn.Linear(
#                 embedding_dim,
#                 hidden_dim,
#             ),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),

#             nn.Linear(
#                 hidden_dim,
#                 hidden_dim,
#             ),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),

#             nn.Linear(
#                 hidden_dim,
#                 hidden_dim // 2,
#             ),
#             nn.BatchNorm1d(hidden_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),

#             nn.Linear(
#                 hidden_dim // 2,
#                 hidden_dim // 4,
#             ),
#             nn.ReLU(inplace=True),

#             nn.Linear(
#                 hidden_dim // 4,
#                 num_classes,
#             ),
#         )

#     def forward(self, x):
#         return self.network(x)
