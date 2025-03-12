import torch

tensor_1 = torch.tensor([[1.0,2,3,4],
                         [5,6,7,8]
                         ])
# tensor_2 = torch.tensor([[8,7],[6,5],[4,3],[2,1]],dtype=torch.float32)

mask = tensor_1 > 3
print("index > 3:",mask)
filtered_tensor = tensor_1[mask]
print("filtered_tensor:",filtered_tensor)
print("filtered_tensor_size:",filtered_tensor.shape)
