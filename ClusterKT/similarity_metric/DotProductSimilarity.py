class DotProductSimilarity(nn.Module):
 
    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output
 
    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result
