# check.py
import torch

print("torch:", torch.__version__)
print("device:",
      "cuda" if torch.cuda.is_available()
      else "mps" if torch.backends.mps.is_available()
      else "cpu")

t = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
print(t @ t.T)   # matrix multiply
