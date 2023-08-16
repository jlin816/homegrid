from collections import defaultdict, Counter
from enum import Enum
import os
import random
from typing import Dict, List
import pathlib
import pickle

import gym
from gym import spaces
import numpy as np
from tokenizers import Tokenizer

from homegrid.base import Pickable, Storage
from homegrid.layout import room2name

class MultitaskWrapper(gym.Wrapper):
  """Continually sample tasks during an episode, rewarding the agent for
  completion."""

  Tasks = Enum("Tasks", [
    "find", "get", "cleanup", "rearrange", "open"],
    start=0)

  def __init__(self, env):
    super().__init__(env)
    self.tasks = list(MultitaskWrapper.Tasks)

  def sample_task(self):
    task_type = random.choice(self.tasks)

    if task_type == MultitaskWrapper.Tasks.find:
      obj_name = random.choice(self.env.objs).name
      task = f"find the {obj_name}"
      def reward_fn(symbolic_state):
        return int(symbolic_state["front_obj"] == obj_name)
    elif task_type == MultitaskWrapper.Tasks.get:
      obj_name = random.choice([ob for ob in self.env.objs if \
                                isinstance(ob, Pickable)]).name
      task = f"get the {obj_name}"
      def reward_fn(symbolic_state):
        return int(symbolic_state["agent"]["carrying"] == obj_name)
    elif task_type == MultitaskWrapper.Tasks.open:
      obj_name = random.choice([ob for ob in self.env.objs if \
          isinstance(ob, Storage)]).name
      task = f"open the {obj_name}"
      def reward_fn(symbolic_state):
        for obj in symbolic_state["objects"]:
          if obj["name"] == obj_name:
            return int(obj["state"] == "open")
    elif task_type == MultitaskWrapper.Tasks.cleanup:
      obj_name = random.choice([ob for ob in self.env.objs if \
                                isinstance(ob, Pickable)]).name
      bin_name = random.choice([ob for ob in self.env.objs if \
                                isinstance(ob, Storage)]).name
      task = f"put the {obj_name} in the {bin_name}"
      def reward_fn(symbolic_state):
        if symbolic_state["agent"]["carrying"] == obj_name:
          return 0.5
        for obj in symbolic_state["objects"]:
          if obj["name"] == bin_name:
            return int(obj_name in obj["contains"])
    elif task_type == MultitaskWrapper.Tasks.rearrange:
      room_code = random.choice(list(self.env.room_to_cells.keys()))
      obj_name = random.choice([ob for ob in self.env.objs if \
                                isinstance(ob, Pickable)]).name
      task = f"move the {obj_name} to the {room2name[room_code]}"
      def reward_fn(symbolic_state):
        if symbolic_state["agent"]["carrying"] == obj_name:
          return 0.5
        for obj in symbolic_state["objects"]:
          if obj["name"] == obj_name:
            return int(obj["room"] == room_code)
    else:
        raise ValueError(f"Unknown task type {task_type}")
    def dist_goal(symbolic_state):
      goal_name = obj_name
      if task_type == MultitaskWrapper.Tasks.cleanup:
        goal_name = bin_name if symbolic_state["agent"]["carrying"] == obj_name \
            else obj_name
      pos = [o for o in symbolic_state["objects"] if \
          o["name"] == goal_name][0]["pos"]
      return abs(self.agent_pos[0] - pos[0]) + abs(self.agent_pos[1] - pos[1])

    self.task = task
    self.reward_fn = reward_fn
    self.dist_goal = dist_goal
    self.subtask_done = False
    self.start_step = self.step_cnt

  def reset(self):
    obs, info = self.env.reset()
    self.step_cnt = 0
    self.start_step = 0
    self.accomplished_tasks = []
    self.task_times = []
    self.sample_task()
    info.update({
        "log_timesteps_with_task": self.step_cnt - self.start_step,
        "log_new_task": True,
        "log_dist_goal": self.dist_goal(info["symbolic_state"])
        })
    return obs, info

  def step(self, action):
    self.step_cnt += 1
    obs, rew, term, trunc, info = self.env.step(action)
    info.update({
        "log_timesteps_with_task": self.step_cnt - self.start_step,
        "log_new_task": False,
        "log_dist_goal": self.dist_goal(info["symbolic_state"])
        })
    if term:
      return obs, rew, term, trunc, info
    rew = self.reward_fn(info["symbolic_state"])
    if rew == 1:
      self.accomplished_tasks.append(self.task)
      self.task_times.append(self.step_cnt - self.start_step)
      self.sample_task()
      info.update({
        "log_timesteps_with_task": self.step_cnt - self.start_step,
        "log_accomplished_tasks": self.accomplished_tasks,
        "log_task_times": self.task_times,
        "log_new_task": True,
        "log_dist_goal": self.dist_goal(info["symbolic_state"])
      })
    elif rew == 0.5:
      if self.subtask_done: rew = 0 # don't reward twice
      self.subtask_done = True
    return obs, rew, term, trunc, info

class LanguageWrapper(gym.Wrapper):
  """Provide the agent with language information one token at a time, using underlying
  environment state and task wrapper.

  Configures types of language available, and specifies logic for which language is provided at
  a given step, if multiple strings are available."""

  def __init__(self,
      env,
      # Max # tokens during prereading phase (for future/dynamics)
      preread_max=-1,
      # How often to repeat the task description
      repeat_task_every=20,
      # Prob of sampling descriptions when we don't have task language
      p_language=0.2,
      debug=False,
      lang_types=["task", "future", "dynamics", "corrections", "termination"],
    ):
    super().__init__(env)
    assert len(lang_types) >= 1 and "task" in lang_types, \
        f"Must have task language, {lang_types}"
    for t in lang_types:
      assert t in ["task", "future", "dynamics", "corrections", "termination"], \
          f"Unknown language type {t}"

    if "dynamics" in lang_types or "future" in lang_types:
      assert preread_max > -1, \
          "Must have preread for dynamics/future language"

    self.instruction_only = len(lang_types) == 1 and lang_types[0] == "task"
    self.preread_max = preread_max
    self.repeat_task_every = repeat_task_every
    self.p_language = p_language
    self.debug = debug
    self.lang_types = lang_types
    self.preread = -1 if self.instruction_only else self.preread_max

    directory = pathlib.Path(__file__).resolve().parent
    with open(directory / "homegrid_embeds.pkl", "rb") as f:
      self.cache, self.embed_cache = pickle.load(f)
    self.empty_token = self.cache["<pad>"]
    # List of tokens of current utterance we're streaming
    self.tokens = [self.empty_token]
    # Index of self.tokens for current timestep
    self.cur_token = 0
    self.embed_size = 512
    self.observation_space = spaces.Dict({
      **self.env.observation_space.spaces,
      "token": spaces.Box(
        0, 32100,
        shape=(),
        dtype=np.uint32),
      "token_embed": spaces.Box(
        -np.inf, np.inf,
        shape=(self.embed_size,),
        dtype=np.float32),
      "is_read_step": spaces.Box(
        low=np.array(False),
        high=np.array(True),
        shape=(),
        dtype=bool),
      "log_language_info": spaces.Text(
        max_length=10000,
      ),
    })
    if self.debug:
      self.tok = Tokenizer.from_pretrained("t5-small")

  def get_descriptions(self, state):
    # facts:
    # - object locations (beginning only but also anytime)
    # - irreversible state (don't change)
    # - dynamics (don't change)
    descs = []
    for obj in state["objects"]:
      if "dynamics" in self.lang_types and obj["action"]:
        descs.append(f"{obj['action']} to open the {obj['name']}")
      if "future" in self.lang_types and obj["room"]:
        descs.append(f"{obj['name']} is in the {room2name[obj['room']]}")
    return descs

  def _tokenize(self, string):
    if string in self.cache:
      return self.cache[string]
    if self.debug:
      return self.tok(string, add_special_tokens=False)["input_ids"]
    raise NotImplementedError(f"tokenize, string not preembedded: >{string}<")

  def _embed(self, string):
    if string in self.embed_cache:
      return self.embed_cache[string]
    if self.debug:
      return [5555] * len(self.tokens)
    raise NotImplementedError(f"embed, string not preembedded: >{string}<")

  def _set_current_string(self, string_or_strings):
    if isinstance(string_or_strings, list):
      self.string = " ".join(string_or_strings)
      self.tokens = [x for string in string_or_strings \
                     for x in self._tokenize(string)]
      self.token_embeds = [x for string in string_or_strings \
                           for x in self._embed(string)]
      self.cur_token = 0
    elif isinstance(string_or_strings, str):
      string = string_or_strings
      self.string = string
      self.tokens = self._tokenize(string)
      self.token_embeds = self._embed(string)
      self.cur_token = 0

  def _increment_token(self):
    if self._lang_is_empty():
      return
    self.cur_token += 1
    if self.cur_token == len(self.tokens):
      self.string = "<pad>"
      self.tokens = [self.empty_token]
      self.token_embeds = [self._embed(self.string)]
      self.cur_token = 0

  def _lang_is_empty(self):
    return self.string == "<pad>"

  def add_language_to_obs(self, obs, info):
    """Adds language keys to the observation:
    - token (int): current token
    - token_embed (np.array): embedding of current token
    - log_language_info (str): human-readable info about language

    On each step, either
      describe new task (will interrupt other language)
      continue tokens that are currently being streamed
      repeat task if it's time
      describe something that changed or will happen (events)
      describe a fact (if not preread) - TODO
      correct the agent - TODO
    """
    if self._step_cnt >= self.preread and info["log_new_task"]:
      # on t=self._step_cnt, we will start streaming the new task
      self._set_current_string(self.env.task)
      self._last_task_repeat = self._step_cnt

    if self._lang_is_empty():
      describable_evts = [e for e in info["events"]
          if e.get("type", "none") in self.lang_types]
      if self.repeat_task_every > 0 and \
          self._step_cnt - self._last_task_repeat >= self.repeat_task_every:
        self._set_current_string(self.env.task)
        self._last_task_repeat = self._step_cnt
      elif len(describable_evts) > 0:
        evt = random.choice(describable_evts)
        self._set_current_string(evt["description"])
      elif np.random.rand() < self.p_language:
        if "corrections" in self.lang_types and \
            info["log_dist_goal"] > self.last_dist:
          self._set_current_string("no, turn around")
        else:
          descs = self.get_descriptions(info["symbolic_state"])
          if len(descs) > 0:
            self._set_current_string(random.choice(descs))

    obs.update({
        "token": self.tokens[self.cur_token],
        "token_embed": self.token_embeds[self.cur_token],
        "log_language_info": self.string,
      })
    self._increment_token()
    return obs

  def reset(self):
    obs, info = self.env.reset()
    obs["is_read_step"] = False
    self.last_dist = info["log_dist_goal"]
    if self.preread_max > -1:
      descs = self.get_descriptions(info["symbolic_state"])
      random.shuffle(descs)
      self._set_current_string(descs)
      self.preread = min(len(self.tokens), self.preread_max)
      obs["image"] = obs["image"] // 2
      obs["is_read_step"] = True
      self.init_obs = obs
      self.init_info = info
    self._step_cnt = 0
    self._last_task_repeat = 0
    obs = self.add_language_to_obs(obs, info)
    return obs, info

  def step(self, action):
    self._step_cnt += 1
    if self._step_cnt <= self.preread:
      obs, rew, term, trunc, info = self.init_obs, 0, False, False, self.init_info
      obs["is_read_step"] = True
    else:
      obs, rew, term, trunc, info = self.env.step(action)
      obs["is_read_step"] = False
    obs = self.add_language_to_obs(obs, info)
    self.last_dist = info["log_dist_goal"]
    return obs, rew, term, trunc, info
