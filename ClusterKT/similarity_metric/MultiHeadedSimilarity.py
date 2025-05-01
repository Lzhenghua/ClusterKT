class MultiHeadedSimilarity(nn.Module):
 
    def __init__(self,
                 num_heads,
                 tensor_1_dim,
                 tensor_1_projected_dim=None,
                 tensor_2_dim=None,
                 tensor_2_projected_dim=None,
                 internal_similarity=DotProductSimilarity()):
        super(MultiHeadedSimilarity, self).__init__()
        self.num_heads = num_heads
        self.internal_similarity = internal_similarity
        tensor_1_projected_dim = tensor_1_projected_dim or tensor_1_dim
        tensor_2_dim = tensor_2_dim or tensor_1_dim
        tensor_2_projected_dim = tensor_2_projected_dim or tensor_2_dim
        if tensor_1_projected_dim % num_heads != 0:
            raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
                             % (tensor_1_projected_dim, num_heads))
        if tensor_2_projected_dim % num_heads != 0:
            raise ValueError("Projected dimension not divisible by number of heads: %d, %d"
                             % (tensor_2_projected_dim, num_heads))
        self.tensor_1_projection = nn.Parameter(torch.Tensor(tensor_1_dim, tensor_1_projected_dim))
        self.tensor_2_projection = nn.Parameter(torch.Tensor(tensor_2_dim, tensor_2_projected_dim))
        self.reset_parameters()
 
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.tensor_1_projection)
        torch.nn.init.xavier_uniform_(self.tensor_2_projection)
 
    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.tensor_1_projection)
        projected_tensor_2 = torch.matmul(tensor_2, self.tensor_2_projection)
 
        # Here we split the last dimension of the tensors from (..., projected_dim) to
        # (..., num_heads, projected_dim / num_heads), using tensor.view().
        last_dim_size = projected_tensor_1.size(-1) // self.num_heads
        new_shape = list(projected_tensor_1.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_1 = projected_tensor_1.view(*new_shape)
        last_dim_size = projected_tensor_2.size(-1) // self.num_heads
        new_shape = list(projected_tensor_2.size())[:-1] + [self.num_heads, last_dim_size]
        split_tensor_2 = projected_tensor_2.view(*new_shape)
 
        # And then we pass this off to our internal similarity function. Because the similarity
        # functions don't care what dimension their input has, and only look at the last dimension,
        # we don't need to do anything special here. It will just compute similarity on the
        # projection dimension for each head, returning a tensor of shape (..., num_heads).
        return self.internal_similarity(split_tensor_1, split_tensor_2)
