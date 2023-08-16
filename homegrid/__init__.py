from gym.envs.registration import register
from homegrid.homegrid_base import HomeGridBase
from homegrid.language_wrappers import MultitaskWrapper, LanguageWrapper
from homegrid.wrappers import RGBImgPartialObsWrapper, FilterObsWrapper

import warnings
warnings.filterwarnings("ignore", module="gym.utils.passive_env_checker")
warnings.filterwarnings("ignore", module="gym.spaces.box")

class HomeGrid:

    def __init__(self, lang_types, *args, **kwargs):
        env = HomeGridBase(*args, **kwargs)
        env = RGBImgPartialObsWrapper(env)
        env = FilterObsWrapper(env, ["image"])
        env = MultitaskWrapper(env)
        env = LanguageWrapper(
            env,
            preread_max=28,
            repeat_task_every=20,
            p_language=0.2,
            lang_types=lang_types,
        )
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

register(
    id="homegrid-task",
    entry_point="homegrid:HomeGrid",
    kwargs={"lang_types": ["task"]},
)

register(
    id="homegrid-future",
    entry_point="homegrid:HomeGrid",
    kwargs={"lang_types": ["task", "future"]},
)

register(
    id="homegrid-dynamics",
    entry_point="homegrid:HomeGrid",
    kwargs={"lang_types": ["task", "dynamics"]},
)

register(
    id="homegrid-corrections",
    entry_point="homegrid:HomeGrid",
    kwargs={"lang_types": ["task", "corrections"]}
)
