import numpy as np
import torch

from robot_pushing_src.robot_pushing_3d import robot_pushing_3d


class RobotPushing3D:

    def __init__(self, n_periods):
        self.n_periods = n_periods
        
    def evaluate(self, X):
        X_unnorm = X.clone()
        for n in range(self.n_periods):
            for i in range(2):
                X_unnorm[:, 3 * n + i] = 10 * X_unnorm[:, 3 * n + i] - 5.0

            X_unnorm[:, 3 * n + 2] =  11.0 * X_unnorm[:, 3 * n + 2] + 1.0

        input_shape = X_unnorm.shape

        output = torch.zeros(input_shape[:-1] + torch.Size([2 * self.n_periods]))

        for i in range(input_shape[0]):
            previous_location = torch.tensor([0.0, 0.0])
            for n in range(self.n_periods):
                np.random.seed(0)
                new_location = torch.tensor(robot_pushing_3d(X_unnorm[i, 3 * n].item(), X_unnorm[i, 3 * n + 1].item(), X_unnorm[i, 3 * n + 2].item(), previous_location[0].item(), previous_location[1].item()))
                output[i, 2 * n : 2 * (n + 1)] = new_location.clone()
                previous_location = new_location.clone()
                
        return output
