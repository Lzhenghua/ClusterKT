class TriLinearSimilarity(nn.Module):
 
    def __init__(self, input_dim, activation=None):
        super(TriLinearSimilarity, self).__init__()
        self.weight_vector = nn.Parameter(torch.Tensor(3 * input_dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.activation = activation
        self.reset_parameters()
 
    def reset_parameters(self):
        std = math.sqrt(6 / (self.weight_vector.size(0) + 1))
        self.weight_vector.data.uniform_(-std, std)
        self.bias.data.fill_(0)
 
    def forward(self, tensor_1, tensor_2):
        combined_tensors = torch.cat([tensor_1, tensor_2, tensor_1 * tensor_2], dim=-1)
        result = torch.matmul(combined_tensors, self.weight_vector) + self.bias
        if self.activation is not None:
            result = self.activation(result)
        return result
