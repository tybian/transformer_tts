import torch

a = torch.FloatTensor([1, 1]).bool()
b = torch.FloatTensor([1, 0]).bool()

c = a | b
print(c)