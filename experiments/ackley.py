import torch
import math

class Ackley:
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.a = 20.
        self.b = 0.2
        self.c = 2 * math.pi
              
    def evaluate(self, X):
        #print(X)
        a = self.a
        b = self.b
        c = self.c
        X_scaled = 4. * X - 2.
        output = torch.zeros(X_scaled.shape[:-1] + torch.Size([3]))
        for i in range(self.input_dim):
            output[..., 0] += X_scaled[..., i] ** 2
            output[..., 1] += torch.cos(c * X_scaled[..., i])
        output /= self.input_dim
        output[..., 2] = a * torch.exp(-b * (torch.sqrt(output[..., 0]))) + torch.exp(output[..., 1]) - a - math.e
        #print(output)   
        return output