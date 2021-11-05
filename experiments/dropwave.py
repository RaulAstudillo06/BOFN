import torch

class Dropwave:
    
    def __init__(self):
        self.n_nodes = 2
        self.input_dim = 2
        
    def evaluate(self, X):
        X_scaled = 10.24 * X - 5.12
        input_shape = X_scaled.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))
        norm_X = torch.norm(X_scaled, dim=-1)
        output[..., 0] = norm_X
        output[..., 1] = (1.0 + torch.cos(12.0 * norm_X)) /(2.0 + 0.5 * (norm_X ** 2))
        return output

