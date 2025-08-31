from train import config
import gymnasium as gym


config["manual_control"] = True
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
