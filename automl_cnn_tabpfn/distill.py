import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def distillation_loss(student_logits, teacher_probs, hard_labels, alpha=0.7, temperature=2.0):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_mlp_head(X_pool, y_hard, teacher_probs, num_classes, epochs=100, lr=1e-3):
    X = torch.tensor(X_pool, dtype=torch.float32)
    y_hard_t = torch.tensor(y_hard, dtype=torch.long)
    y_soft_t = torch.tensor(teacher_probs, dtype=torch.float32)

    model = MLPHead(in_dim=X.shape[1], num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = distillation_loss(logits, y_soft_t, y_hard_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"epoch {epoch}: loss {loss.item():.4f}")

    return model