import gymnasium as gym
import numpy as np
import highway_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

kinematics= {"type": "Kinematics",
             "features": ["x", "vx", "vy"],
             "absolute": False,
             "normalize": True,
             "vehicles_count": 1}
occupancy_grid = {"type": "OccupancyGrid",
        "features": ["presence", "on_road"],
        "grid_size": [[0, 56], [-16, 16]], #  only forward looking
        "grid_step": [2, 2],
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
    "other_vehicles": 10,
    "screen_width": 600,
    "screen_height": 600,
    "centering_position": [0.5, 0.5],
    "speed_limit": 30,
    "terminate_off_road": True,
}

def make_env():
    return  gym.make("racetrack-large-v0", config = config,)

def test_observation_space():
    """Test function to check observation space dimensions"""
    env = make_env()
    obs, _ = env.reset()
    print("Observation space:", env.observation_space)
    print("Sample observation shape:")
    for key, value in obs.items():
        print(f"  {key}: {np.array(value).shape}")
    total_features = sum(np.array(value).size for value in obs.values())
    print(f"Total features: {total_features}")
    env.close()
    return total_features



if __name__ == "__main__":
    test_observation_space()

    n_runs = 10 # number of training runs
    n_timesteps = 1e5 # number of timesteps per training

    cores = 12
    batch_size = 64
    nsteps = 128
    training_data = nsteps * cores
    batch_size = training_data //8
    env = make_vec_env(make_env, n_envs=cores, vec_env_cls=SubprocVecEnv, vec_env_kwargs={'start_method': 'fork'} ) # Subproccesenv does not work something with multiInput

    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=nsteps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=4e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log="racetrack_ppo/",
        device="cpu",)
    # model = PPO.load(path= "models/racecar-3400000", env = env, device='cpu')

    # Train
    iter = 0
    while True:
        iter += 1
        model.learn(total_timesteps=n_timesteps, reset_num_timesteps=False)
        model.save(f"models/racecar2-{int(iter)}.zip")







