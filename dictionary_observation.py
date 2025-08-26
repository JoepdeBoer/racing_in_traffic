import functools
import highway_env.envs.common.observation as observation
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.observation import ObservationType, TimeToCollisionObservation, KinematicObservation, \
    OccupancyGridObservation, KinematicsGoalObservation, GrayscaleObservation, AttributesObservation, \
    MultiAgentObservation, TupleObservation, LidarObservation, ExitObservation
from gymnasium import spaces






old_func = observation.observation_factory

@functools.wraps(old_func)
def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "TupleObservation":
        return TupleObservation(env, **config)
    elif config["type"] == "DictObservation":
        return DictObservation(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "ExitObservation":
        return ExitObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
