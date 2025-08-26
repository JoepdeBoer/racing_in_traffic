import gymnasium as gym
import numpy as np
import highway_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

kinematics= {"type": "Kinematics",
             "features": ["x", "y", "vx", "vy"],
             "absolute": False,
             "normalize": True,
             "vehicles_count": 0}
occupancy_grid = {"type": "OccupancyGrid",
        "features": ["presence", "on_road"],
        "grid_size": [[-18, 18], [0, 36]], #  only forward looking
        "grid_step": [3, 3],
        "as_image": False,
        "align_to_vehicle_axes": True, }

observation = {"type": "DictObservation",
               "observation_configs": [kinematics, occupancy_grid]}


config = {
    "observation": observation,
    "action": {
        "type": "ContinuousAction",
        "longitudinal": True,
        "lateral": True,
        "dynamical": True,
        "steering_range": [-np.pi/3, np.pi/3],
    },
    "simulation_frequency": 22,
    "policy_frequency": 5,
    "duration": 300,
    "collision_reward": -1,
    "lane_centering_cost": 4,
    "lane_centering_reward": 1,
    "action_reward": -0.3,
    "controlled_vehicles": 1,
    "other_vehicles": 5,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "speed_limit": 25,
    "terminate_off_road": True,
}

def make_env():
    return  gym.make("racetrack-large-v0", config = config)



if __name__ == "__main__":
    n_runs = 10 # number of training runs
    n_timesteps = 1e5 # number of timesteps per training

    cores = 6
    batch_size = 64
    env = make_vec_env(make_env, n_envs=cores, vec_env_cls=SubprocVecEnv)

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=batch_size * 12 // cores,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log="racetrack_ppo/",
        device="cpu",)

    # Train

    model.learn(total_timesteps=int(n_timesteps), reset_num_timesteps = True) # TODO set to false when second run
    model.save(f"models/model{0}")







