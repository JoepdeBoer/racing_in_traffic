import gymnasium as gym
import numpy as np
import highway_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO  # Remove PPO2 import - it doesn't exist in SB3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

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

config = {
    "observation": observation,
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
        "dynamical": True,
        "steering_range": [-np.pi / 3, np.pi / 3],
    },
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -1,
    "lane_centering_cost": 4,
    "lane_centering_reward": 0.3,
    "right_lane_reward": 0.6,
    "action_reward": -0.7,
    "speed_reward": 1,
    "controlled_vehicles": 1,
    "other_vehicles": 2,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "speed_limit": 7,
    "average_speed": 5,
    "terminate_off_road": True,
}


def make_env():
    return gym.make("racetrack-large-v0", config=config, max_episode_steps=512)


def test_observation_space():
    """Test function to check observation space dimensions"""
    env = make_env()
    obs, *_ = env.reset()  # Fixed unpacking
    print("Observation space:", env.observation_space)  # Fixed typo
    print("Sample observation shape:")
    for key, value in obs.items():
        print(f"  {key}: {np.array(value).shape}")
    total_features = sum(np.array(value).size for value in obs.values())
    print(f"Total features: {total_features}")
    env.close()
    return total_features


if __name__ == "__main__":  # Fixed syntax
    test_observation_space()

    n_runs = 10  # number of training runs
    n_timesteps = int(1e5)  # Convert to int for cleaner code

    cores = 12
    nsteps = 512
    training_data = nsteps * cores
    batch_size = training_data // 8

    env = make_vec_env(
        make_env,
        n_envs=cores,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={'start_method': 'fork'}
    )

    # model = PPO(
    #     "MultiInputPolicy",
    #     env,
    #     policy_kwargs=dict(net_arch=dict(pi=[1024, 512, 256], vf=[1024, 512, 256])),
    #     n_steps=nsteps,
    #     batch_size=batch_size,
    #     n_epochs=10,
    #     learning_rate=3e-4,
    #     gamma=0.99,  # Increased from 0.9 for better long-term rewards
    #     gae_lambda=0.95,  # Added GAE lambda for more stable advantage estimation
    #     clip_range=0.2,  # Standard PPO clipping
    #     ent_coef=0.01,  # Small entropy coefficient to encourage exploration
    #     vf_coef=0.5,  # Value function coefficient
    #     max_grad_norm=0.5,  # Gradient clipping for stability
    #     verbose=2,
    #     tensorboard_log="bigbrainlimo_ppo/",
    #     device="cpu",
    # )


    new_model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512])),
        n_steps=nsteps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,  # Increased from 0.9 for better long-term rewards
        gae_lambda=0.95,  # Added GAE lambda for more stable advantage estimation
        clip_range=0.2,
        # ent_coef=0.01,  # Small entropy coefficient to encourage exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping for stability
        verbose=2,
        tensorboard_log="small2limo_ppo/",
        device="cuda",
    )
    # new_model.policy.load_state_dict(old_model.policy.state_dict())

    # Training loop
    iter = 0  # Start from 0 if creating new model
    while True:
        iter += 1
        print(f"Starting training iteration {iter}")
        new_model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
        new_model.save(f"models/small2vip-{iter}")
        print(f"Model saved as stable2vip-{iter}")

        # Optional: Add a break condition to avoid infinite training
