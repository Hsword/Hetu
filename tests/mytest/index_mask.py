import torch

a=torch.arange(80).reshape((4, 20))

index = torch.tensor([3, 3, 3, 2, 2, 2, 1, 1, 1, 2, 1, 0, 2, 1, 0, 3, 0, 0, 0])

mask = torch.tensor([False,  True,  True,  True,  True,  True,  True,  True,  True,  True, True,  True,  True,  True,  True,  True,  True,  True,  True,  True])


print(a.shape)
print(a)
print(index.shape)
print(index)
print(mask.shape)
print(mask)

print("a[index, mask]:")
print(a[index, mask])
