import torch
import torch.nn as nn

class MyMLP(nn.Module):
    """
    MLP สำหรับ XOR / 2D Binary Classification
    Input(2) -> Hidden(8) -> ReLU -> Output(1) -> Sigmoid
    """
    def __init__(self, in_features=2, hidden_dim=8, out_features=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class BreastCancerMLP(nn.Module):
    """
    MLP สำหรับ Breast Cancer Classification (30 Features)
    Input 30 -> Hidden(16) -> ReLU -> Hidden(8) -> ReLU -> Output(1) -> Sigmoid
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


if __name__ == '__main__':
    mlp_xor = MyMLP()
    print("XOR Model Architecture:")
    print(mlp_xor)
    print(f"Total Parameters: {sum(p.numel() for p in mlp_xor.parameters())}\n")

    mlp_bc = BreastCancerMLP()
    print("Breast Cancer Model Architecture:")
    print(mlp_bc)
    print(f"Total Parameters: {sum(p.numel() for p in mlp_bc.parameters())}")
