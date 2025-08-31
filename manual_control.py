from train import config
import gymnasium as gym
kinematics = {"type": "Kinematics",
              "features": ["x", "vx", "vy"],
              "absolute": False,
              "normalize": True,
              "vehicles_count": 1}

occupancy_grid = {"type": "OccupancyGrid",
                  "features": ["on_road"],
                  "grid_size": [[0, 48], [-16, 16]],
                  "grid_step": [3, 3],
                  "as_image": False,
                  "align_to_vehicle_axes": True, }

Lidar = {"type": "LidarObservation",
         "cells": 64,
         "maximum_range": 64,
         "normalize": True,
         }

observation = {"type": "DictObservation",
               "observation_configs": [kinematics, Lidar, occupancy_grid]}

config["manual_control"] = True
config["observation"] = observation
# config["controlled_vehicles"] = 1
# config["other_vehicles"] = 2
# config["average_speed"] = 6
# config["speed_limit"] = 10
# config['simulation_frequency'] = 15
# config['policy_frequency'] = 5

# config["screen_width"] = 1200
# config["screen_height"] = 900
config['terminate_off_road'] = False
env = gym.make("racetrack-large", config=config, render_mode = "human")
env.reset()
done = False

while not done:
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())  # with manual control, these actions are ignored
    done = terminated or truncated
env.close()
