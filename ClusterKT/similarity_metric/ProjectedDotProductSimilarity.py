class ProjectedDotProductSimilarity(nn.Module):
   
    def __init__(self, tensor_1_dim, tensor_2_dim, projected_dim,
                 reuse_weight=False, bias=False, activation=None):
        super(ProjectedDotProductSimilarity, self).__init__()
        self.reuse_weight = reuse_weight
        self.projecting_weight_1 = nn.Parameter(torch.Tensor(tensor_1_dim, projected_dim))
        if self.reuse_weight:
            if tensor_1_dim != tensor_2_dim:
                raise ValueError('if reuse_weight=True, tensor_1_dim must equal tensor_2_dim')
        else:
            self.projecting_weight_2 = nn.Parameter(torch.Tensor(tensor_2_dim, projected_dim))
        self.bias = nn.Parameter(torch.Tensor(1)) if bias else None
        self.activation = activation
 
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.projecting_weight_1)
        if not self.reuse_weight:
            nn.init.xavier_uniform_(self.projecting_weight_2)
        if self.bias is not None:
            self.bias.data.fill_(0)
 
    def forward(self, tensor_1, tensor_2):
        projected_tensor_1 = torch.matmul(tensor_1, self.projecting_weight_1)
        if self.reuse_weight:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_1)
        else:
            projected_tensor_2 = torch.matmul(tensor_2, self.projecting_weight_2)
        result = (projected_tensor_1 * projected_tensor_2).sum(dim=-1)
        if self.bias is not None:
            result = result + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result
