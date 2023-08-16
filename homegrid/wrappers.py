from gym.core import Wrapper, ObservationWrapper
import gym
from gym import spaces
from typing import List, Union
import numpy as np

class Gym26Wrapper(Wrapper):
    """Wraps gym v0.26 env with a ~v0.22 API so it can be used with sb3.

    Refer to gym wrappers for opposite compatibility (old env -> new API):
    https://github.com/openai/gym/blob/master/gym/wrappers/compatibility.py
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, mode):
        return self.env.render()

class FilterObsWrapper(ObservationWrapper):

    def __init__(self, env, obs_keys: List[str]):
        super().__init__(env)
        self.obs_keys = obs_keys
        self.observation_space = spaces.Dict({
            k: v for k, v in env.observation_space.items() if k in self.obs_keys
        })

    def observation(self, obs):
        return {k: v for k, v in obs.items() if k in self.obs_keys}

class RGBImgPartialObsWrapper(ObservationWrapper):
    """RGBImg wrapper that also preserves the original symbolic observation."""

    def __init__(self, env, tile_size=32):
        super().__init__(env)

        # Rendering attributes for observations
        self.tile_size = tile_size

        obs_shape = env.observation_space.spaces["image"].shape
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0] * tile_size, obs_shape[1] * tile_size, 3),
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img_partial = self.get_frame(tile_size=self.tile_size, agent_pov=True)

        return {**obs, "symbolic_image": obs["image"], "image": rgb_img_partial}
