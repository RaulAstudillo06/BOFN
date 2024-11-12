import numpy as np
import os
import torch

from gym.envs.robotics.fetch_env import  FetchEnv


class FetchReach:

    def __init__(self, n_periods):
        self.n_periods = n_periods
        self.origin = [1.34183679, 0.74910472, 0.41363141]
        self.model_xml_path = os.path.join("fetch", "reach.xml")
        
    def evaluate(self, X):
        X_unnorm = 2.0 * X - 1.0
        input_shape = X_unnorm.shape

        output = []

        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }

        for i in range(input_shape[0]):
            env = FetchEnv(
                    self.model_xml_path,
                    has_object=False,
                    block_gripper=True,
                    n_substeps=100,
                    gripper_extra_height=0.0,
                    target_in_the_air=True,
                    target_offset=0.0,
                    obj_range=0.0,
                    target_range=0.0,
                    distance_threshold=0.05,
                    initial_qpos=initial_qpos,
                    reward_type="sparse",
                )
            env.reset()
            env.seed(0)
            output.append([])
            for n in range(self.n_periods):
                action = np.asanyarray([X_unnorm[i, 3 * n], X_unnorm[i, 3 * n + 1], X_unnorm[i, 3 * n + 2], 0.0])
                # Take same action twice
                #env.step(action)
                obs, reward, done, info = env.step(action)
                #print(obs)
                for d in range(3):
                    output[i].append(obs["achieved_goal"][d] - self.origin[d])
            
            env.close()

        output = 100 * torch.tensor(output)  
        return output
