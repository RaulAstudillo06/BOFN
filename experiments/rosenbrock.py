import torch

class Rosenbrock:
    
    def __init__(self, dim):
        self.dim = dim
        self.n_nodes = dim - 1
        
    def evaluate(self, X):
        X_scaled = 4.0 * X - 2.0
        batch_size = X_scaled.shape[0]
        output = torch.empty((batch_size, self.n_nodes))
        output[..., 0] = 100.0 *  (X_scaled[..., 1] - X_scaled[..., 0] ** 2) ** 2  + (X_scaled[..., 0] - 1.0) ** 2
        for i in range(1, self.n_nodes):
            output[..., i] = 100.0 *  (X_scaled[..., i + 1] - X_scaled[..., i] ** 2) ** 2  + (X_scaled[..., i] - 1.0) ** 2
            output[..., i] += output[..., i - 1]
        output = -output
        return output

