import torch

class Alpine2:
    
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        
    def evaluate(self, X):
        X_scaled = 10 * X
        input_shape = X_scaled.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))
        output[..., 0] = torch.sqrt(X_scaled[..., 0]) * torch.sin(X_scaled[..., 0])
        for k in range(1, self.n_nodes):
            output[..., k] = torch.sqrt(X_scaled[..., k]) * torch.sin(X_scaled[..., k]) * output[..., k - 1]
            
        return output