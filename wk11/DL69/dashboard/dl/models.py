import torch
import torch.nn as nn

class MyMLP(nn.Module):
    """
    MLP สำหรับ Breast Cancer Classification (การบ้านสัปดาห์ที่ 10)
    Input 30 -> Hidden(16) -> ReLU -> Hidden(8) -> ReLU -> Output(1) -> Sigmoid
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(30, 16)
        self.relu1  = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2  = nn.ReLU()
        self.layer3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x
