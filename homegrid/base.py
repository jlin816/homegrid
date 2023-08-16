import hashlib
import math
from abc import abstractmethod
from enum import IntEnum
from typing import Any, Callable, Optional, Union
import random
import os

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from PIL import Image

# Size in pixels of a tile in the full-scale human view
from homegrid.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
    draw_obj,
)
from homegrid.window import Window

TILE_PIXELS = 32
SHOW_GRIDLINES = False
# Draw robot instead of default triangle for agent
USE_AGENT_TEXTURE = True
if USE_AGENT_TEXTURE:
    AGENT_TEXTURE = np.asarray(Image.open(
        f"{os.path.dirname(__file__)}/assets/robot.png"
    ))
# Center the agent in the view
CENTERED_VIEW = True
# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, name):
        self.name = name
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

    def tick(self):
        pass

    def encode(self):
        """Tuple encoding of this object for render caching."""
        return (self.name,)


## Homegridv2
class Storage(WorldObj):

    def __init__(self, name, textures, state=None, action=None,
        contains=None, reset_broken_after=20):
        super().__init__(name.replace("_", " "))
        self.textures = {
            **textures,
            "broken": np.rot90(textures["closed"])}
        self.contains = contains or []
        # valid states {"open", "closed", "broken"}
        self.state = state if state else \
            random.choice(["open", "closed"])
        self.action = action if action else \
            random.choice(["pedal", "grasp", "lift"])
        self.broken_t = 0
        self.reset_broken_after = reset_broken_after

    def agent_can_overlap(self):
        return False

    def can_overlap(self):
        return True

    def render(self, img):
        draw_obj(img, self.textures[self.state])

    def encode(self):
        return (self.name, self.state)

    def _get_contents(self):
        if len(self.contains) == 0:
          return None
        return self.contains.pop()

    def interact(self, action, obj=None):
        if action == "get":
          if self.state != "open":
            return False
          return self._get_contents()
        elif action == "drop":
          if self.state != "open":
            return False
          if len(self.contains) == 0 and obj:
            obj.cur_pos = (-1, -1)
            self.contains.append(obj)
            return True
        elif action in {"pedal", "grasp", "lift"}:
          if self.state == "closed":
            if action != self.action:
              self.state = "broken"
              self.broken_t = 0
            else:
              self.state = "open"
              return True
        else:
          raise NotImplementedError(f"Attempting to interact with {action}")
        return False

    def tick(self):
        if self.state == "broken":
            self.broken_t += 1
        if self.broken_t == self.reset_broken_after:
            self.state = "closed"
            self.broken_t = 0


class Pickable(WorldObj):

    def __init__(self, name, texture, invisible=False):
        super().__init__(name.replace("_", " "))
        self.texture = texture
        self.invisible = invisible
        # If invisible, make visible after N steps
        self.invisible_count = 5

    def agent_can_overlap(self):
        return self.invisible

    def can_overlap(self):
        return True

    def can_pickup(self):
        return True

    def render(self, img):
        if self.invisible: return
        draw_obj(img, self.texture)

    def encode(self):
        return (self.name, self.invisible)

    def tick(self):
      if self.invisible:
        self.invisible_count -= 1
      if self.invisible_count == 0:
        self.invisible = False


class Inanimate(WorldObj):

    def __init__(self, name, texture, can_overlap=False):
        super().__init__(name)
        self.texture = texture
        self._can_overlap = can_overlap

    def agent_can_overlap(self):
        return False

    def can_overlap(self):
        return self._can_overlap

    def render(self, img):
        draw_obj(img, self.texture)

    def encode(self):
        return (self.name,)


class FloorWithObject(WorldObj):

    def __init__(self, name, texture, agent_can_overlap,
                 can_overlap):
        super().__init__(name)
        self.texture = texture
        self._agent_can_overlap = agent_can_overlap
        self._can_overlap = can_overlap

    def agent_can_overlap(self):
        return self._agent_can_overlap

    def can_overlap(self):
        return self._can_overlap

    def render(self, img):
        draw_obj(img, self.texture)

    def encode(self):
        return (self.name,)


class Wall(WorldObj):
    def __init__(self):
        super().__init__("wall")

    def agent_can_overlap(self):
        return False

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1),
                    np.array([100, 100, 100]))


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height
        self.floor_grid = [None] * width * height

#    def __contains__(self, key):
#        if isinstance(key, WorldObj):
#            for e in self.grid:
#                if e is key:
#                    return True
#        elif isinstance(key, tuple):
#            for e in self.grid:
#                if e is None:
#                    continue
#                if (e.color, e.type) == key:
#                    return True
#                if key[0] is None and key[1] == e.type:
#                    return True
#        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy

        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def set_floor(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.floor_grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def get_floor(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.floor_grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                floor = self.get_floor(i, j)
                grid.set(j, grid.height - 1 - i, v)
                grid.set_floor(j, grid.height - 1 - i, floor)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and y >= 0 and y < self.height:
                    v = self.get(x, y)
                    floor = self.get_floor(x, y)
                else:
                    v = Wall()
                    floor = None

                grid.set(i, j, v)
                grid.set_floor(i, j, floor)

        return grid

    @classmethod
    def render_tile(
        cls, obj, floor, agent_dir=None, highlight=False,
        tile_size=TILE_PIXELS, subdivs=3, bgcolor="white", pov_dir=None):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = floor.encode() + key if floor else key
        key = obj.encode() + key if obj else key
        # rotate objects depending on agent perspective
        key = (pov_dir,) + key if pov_dir else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        # Render floor tile
        if bgcolor == "black":
            img = np.zeros(
                shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
            )
        elif bgcolor == "white":
            img = np.ones(
                shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
            ) * 255
        else:
            raise ValueError("Unknown color")

        if floor is not None:
            floor.render(img)

        # Draw the grid lines (top and left edges)
        if SHOW_GRIDLINES:
            fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
            fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Rotate object textures depending on agent perspective
        if not CENTERED_VIEW:
          if pov_dir is not None:
              img = np.rot90(img, k=(pov_dir + 1) % 4)

        # Overlay the agent on top
        if agent_dir is not None:
            if USE_AGENT_TEXTURE:
              draw_obj(img, AGENT_TEXTURE)
              # Show direction indicator
              if CENTERED_VIEW:
                tri_fn = point_in_triangle(
                    (0.65, 0.29),
                    (0.87, 0.50),
                    (0.65, 0.71),
                )
                tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
                fill_coords(img, tri_fn, (255, 0, 0))
            else:
              tri_fn = point_in_triangle(
                  (0.12, 0.19),
                  (0.87, 0.50),
                  (0.12, 0.81),
              )

              # Rotate the agent based on its direction
              tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
              fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(self, tile_size, agent_pos, agent_dir=None, highlight_mask=None,
               pov_dir=None):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                floor = self.floor_grid[j * self.width + i]

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    floor,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    pov_dir=pov_dir,
                    subdivs=1,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def process_vis(self, agent_pos):
        mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, self.height)):
            for i in range(0, self.width - 1):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, self.width)):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, self.height):
            for i in range(0, self.width):
                if not mask[i, j]:
                    self.set(i, j, None)
                    self.set_floor(i, j, None)

        return mask


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2
        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

        # Done completing task
        done = 6

    def __init__(
        self,
        grid_size: int = None,
        width: int = None,
        height: int = None,
        max_steps: int = 100,
        see_through_walls: bool = True,
        agent_view_size: int = 7,
        render_mode: Optional[str] = None,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width is None and height is None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
            }
        )

        # Range of possible rewards
        self.reward_range = (0, 1)

        self.window: Window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos: np.ndarray = None
        self.agent_dir: int = None

        # Current grid and mission and carrying
        self.grid = Grid(width, height)
        self.carrying = None

        # Rendering attributes
        self.render_mode = render_mode
        self.highlight = highlight
        self.tile_size = tile_size
        self.agent_pov = agent_pov

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reinitialize episode-specific variables
        self.agent_pos = (-1, -1)
        self.agent_dir = -1

        # Generate a new random grid at the start of each episode
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert (
            self.agent_pos >= (0, 0)
            if isinstance(self.agent_pos, tuple)
            else all(self.agent_pos >= 0) and self.agent_dir >= 0
        )

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self._init_inventory()

        # Step count since episode start
        self.step_count = 0

        if self.render_mode == "human":
            self.render()

        # Return first observation
        obs = self.gen_obs()

        return obs, {}

    def _init_inventory(self):
        self.carrying = None

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode("utf8"))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.integers(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return self.np_random.integers(0, 2) == 0

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(xLow, xHigh),
            self.np_random.integers(yLow, yHigh),
        )

    def place_obj(self, obj, top=None, size=None, reject_fn=None, max_tries=math.inf):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError("rejection sampling failed in place_obj")

            num_tries += 1

            pos = np.array(
                (
                    self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                    self._rand_int(top[1], min(top[1] + size[1], self.grid.height)),
                )
            )

            pos = tuple(pos)

            # Don't place the object on top of another object
            if self.grid.get(*pos) is not None and not self.grid.get(*pos).can_overlap():
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(pos[0], pos[1], obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = (-1, -1)
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = rx * lx + ry * ly
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        """

        agent_view_size = agent_view_size or self.agent_view_size

        if not CENTERED_VIEW:
          # Facing right
          if self.agent_dir == 0:
              topX = self.agent_pos[0]
              topY = self.agent_pos[1] - agent_view_size // 2
          # Facing down
          elif self.agent_dir == 1:
              topX = self.agent_pos[0] - agent_view_size // 2
              topY = self.agent_pos[1]
          # Facing left
          elif self.agent_dir == 2:
              topX = self.agent_pos[0] - agent_view_size + 1
              topY = self.agent_pos[1] - agent_view_size // 2
          # Facing up
          elif self.agent_dir == 3:
              topX = self.agent_pos[0] - agent_view_size // 2
              topY = self.agent_pos[1] - agent_view_size + 1
          else:
              assert False, "invalid agent direction"

          botX = topX + agent_view_size
          botY = topY + agent_view_size
        else:
          topX = self.agent_pos[0] - 1
          topY = self.agent_pos[1] - 1
          botX = self.agent_pos[0] + 2
          botY = self.agent_pos[1] + 2
        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def step(self, action):
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        if not CENTERED_VIEW:
            for i in range(self.agent_dir + 1):
                grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        if CENTERED_VIEW:
          agent_pos = grid.width // 2, grid.height // 2
        else:
          agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        # image = grid.encode(vis_mask)
        image = None

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir}

        return obs

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid()

        agent_pos = (self.agent_view_size // 2, self.agent_view_size // 2) if \
          CENTERED_VIEW else (self.agent_view_size // 2, self.agent_view_size - 1)
        agent_dir = self.agent_dir if CENTERED_VIEW else 3

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            highlight_mask=vis_mask,
            pov_dir=self.agent_dir,
        )

        return img

    def get_full_render(self, highlight, tile_size):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        if CENTERED_VIEW:
          for abs_i in range(self.agent_pos[0] - 1, self.agent_pos[0] + 2):
            for abs_j in range(self.agent_pos[1] - 1, self.agent_pos[1] + 2):
              highlight_mask[abs_i, abs_j] = True
        else:
          f_vec = self.dir_vec
          r_vec = self.right_vec
          top_left = (
              self.agent_pos
              + f_vec * (self.agent_view_size - 1)
              - r_vec * (self.agent_view_size // 2)
          )

          # For each cell in the visibility mask
          for vis_j in range(0, self.agent_view_size):
              for vis_i in range(0, self.agent_view_size):
                  # If this cell is not visible, don't highlight it
                  if not vis_mask[vis_i, vis_j]:
                      continue

                  # Compute the world coordinates of this cell
                  abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                  if abs_i < 0 or abs_i >= self.width:
                      continue
                  if abs_j < 0 or abs_j >= self.height:
                      continue

                  # Mark this cell to be highlighted
                  highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
        )

        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = TILE_PIXELS,
        agent_pov: bool = False,
    ):
        """Returns an RGB image corresponding to the whole environment or the agent's point of view.

        Args:

            highlight (bool): If true, the agent's field of view or point of view is highlighted with a lighter gray color.
            tile_size (int): How many pixels will form a tile from the NxM grid.
            agent_pov (bool): If true, the rendered frame will only contain the point of view of the agent.

        Returns:

            frame (np.ndarray): A frame of type numpy.ndarray with shape (x, y, 3) representing RGB values for the x-by-y pixel image.

        """
        if agent_pov:
            return self.get_pov_render(tile_size)
        else:
            return self.get_full_render(highlight, tile_size)

    def render(self):

        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)

        if self.render_mode == "human":
            if self.window is None:
                self.window = Window("minigrid")
                self.window.show(block=False)
            self.window.show_img(img)
        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            self.window.close()
