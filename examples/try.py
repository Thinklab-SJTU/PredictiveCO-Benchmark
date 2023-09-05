import torch

def top_k_one_hot_torch(input_tensor, k):
    _, indices = torch.topk(input_tensor, k=k, dim=-1)
    top_k_one_hot = torch.zeros_like(input_tensor)
    top_k_one_hot.scatter_(-1, indices, 1)
    return top_k_one_hot

# 例子使用
n = 10  # 向量长度
k = 3   # 前k个最大值
input_tensor = torch.tensor([3.0, 1.0, 4.0, 2.0, 5.0, 0.0, 9.0, 7.0, 6.0, 8.0], dtype=torch.float32)
output_vector = top_k_one_hot_torch(input_tensor, k)
print(output_vector.numpy())
