import gymnasium as gym
import highway_env
from highway_env.envs.racetrack_env import RacetrackEnvLarge
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

kinematics= {"type": "Kinematics", }
occupancy_grid = {"type": "OccupancyGrid", }

action = {}


if __name__ == "__main__":
    n_runs = 10 # number of training runs
    n_timesteps = 1e5 # number of timesteps per training


    cores = 12
    batch_size = 64
    env = make_vec_env()
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // cores,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log="racetrack_ppo/")

    # Train
    for i in range(n_runs):
        model.learn(total_timesteps=int(1e5))
        model.save("racetrack_ppo/model")
        del model


