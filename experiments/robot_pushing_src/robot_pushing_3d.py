#!/usr/bin/env python
# Copyright (c) 2017 Zi Wang
from .push_world import *
import sys


def robot_pushing_3d(
    rx: float,
    ry: float,
    duration: float,
    ox: float,
    oy: float,
) -> float:
    simu_steps = int(10 * duration)
    # set it to False if no gui needed
    world = b2WorldInterface(False)
    oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size  = 'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (0.3,1) 
    thing,base = make_thing(500, 500, world, oshape, osize, ofriction, odensity, bfriction, (ox,oy))

    init_angle = np.arctan(ry/rx)
    robot = end_effector(world, (rx,ry), base, init_angle, hand_shape, hand_size)
    final_location = simu_push(world, thing, robot, base, simu_steps)
    return final_location
