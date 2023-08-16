"""Store room layouts and assets."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import numpy as np

from homegrid.base import (
    Wall, Storage, Inanimate, Pickable, FloorWithObject
)
from homegrid.rendering import draw_obj

@dataclass
class RoomSpec:
	name: str
	texture: str


ROOMS = {
	"K": RoomSpec("kitchen", "tile"),
	"L": RoomSpec("living room", "carpet"),
	"D": RoomSpec("dining room", "wood"),
}
room2name = {k: v.name for k, v in ROOMS.items()}
# All static and interactive objects
OBJECTS = [
	"cupboard",
	"stove",
	"fridge",
	"countertop",
	"chairl",
	"chairr",
	"table",
	"sofa",
	"sofa_side",
	"rugl",
	"rugr",
	"coffeetable",
	"cabinet",
	"plant",
]
# Objects that can have things placed on top of them
SURFACES = {
	"cupboard",
	"stove",
	"countertop",
	"chairl",
	"chairr",
	"table",
	"sofa",
	"sofa_side",
	"rugl",
	"rugr",
	"coffeetable",
}
# Trash objects
TRASH = [
	"bottle",
	"fruit",
	"papers",
	"plates",
]
# Trash receptacles
CANS = [
	"recycling_bin",
	"trash_bin",
	"compost_bin",
]


class ThreeRoom:
	LAYOUT = {
		# Room layout, keys into ROOMS
		"rooms":
"""......WWWWWWWW
......WLLLLLLW
......WLLLLLLW
......WLLLLLLW
......WLLLLLLW
WWWWWWWWLLLLLW
WKKKKKKWDDDDDW
WKKKKKKKDDDDDW
WKKKKKKKDDDDDW
WKKKKKKKDDDDDW
WKKKKKKWDDDDDW
WWWWWWWWWWWWWW
""".splitlines(),
		# Static (non-interactive) objects, keys into OBJECTS
		"fixed_objects":
"""......WWWWWWWW
......WnhlhLmW
......WiLLLLLW
......WiLjkLLW
......WLLLLLLW
WWWWWWWWLLLLLW
WaabbacWDDDDDW
WKKKKKKKDDDDDW
WKKKKKKKDegfDW
WKKKKKKKDegfDW
WddddddWDDDDDW
WWWWWWWWWWWWWW
""".splitlines(),
		# Valid positions to place agent and objects
		"valid_poss": {
			"agent_start":
"""..............
..............
........xxxxx.
........xxxxx.
........xxxxx.
........xxxxx.
........xxxxx.
.xxxxxxxxxxxx.
.xxxxxxxx...x.
.xxxxxxxx...x.
........xxxxx.
..............
""".splitlines(),
			"obj":
"""..............
.........x....
........x...x.
..............
.......x......
..............
..x..x........
.............
..........x...
..........x...
..xx..........
..............
""".splitlines(),
			"can":
"""..............
...........x..
..............
..............
..............
..............
..............
..............
..............
.x............
............x.
..............
""".splitlines(),
}}

	def __init__(self):
		self._load_textures()
		self._parse_valid_poss()
		self.width = len(self.LAYOUT["rooms"][0])
		self.height = len(self.LAYOUT["rooms"])
		self.room_to_cells = defaultdict(list)
		self.cell_to_room = {}
		for y in range(len(self.LAYOUT["rooms"])):
			for x in range(len(self.LAYOUT["rooms"][y])):
				if self.LAYOUT["rooms"][y][x] != "." and self.LAYOUT["rooms"][y][x] != "W":
					room_code = self.LAYOUT["rooms"][y][x]
					self.room_to_cells[room_code].append((x, y))
					self.cell_to_room[(x, y)] = room_code

	def _load_textures(self):
		textures = {}
		for fname in Path(__file__).parent.glob("assets/**/*.png"):
			name = fname.stem
			textures[name] = np.asarray(Image.open(fname))
		self.textures = textures

	def _parse_valid_poss(self):
		valid_poss = defaultdict(list)
		for k, grid in self.LAYOUT["valid_poss"].items():
			for y, line in enumerate(grid):
				for x, c in enumerate(line):
					if c == "x":
						valid_poss[k].append((x, y))
		self.valid_poss = valid_poss

	def populate(self, grid):
		"""Fill the grid with the floor and fixed object layout."""
		for y in range(len(self.LAYOUT["rooms"])):
			for x in range(len(self.LAYOUT["rooms"][y])):
				if self.LAYOUT["rooms"][y][x] == ".":
					continue
				elif self.LAYOUT["rooms"][y][x] == 'W':
					grid.set(x, y, Wall())
				else:
					room_code = self.LAYOUT["rooms"][y][x]
					assert room_code in ROOMS.keys(),\
							"Invalid room code: {}".format(room_code)
					floor = ROOMS[room_code].texture
					name = floor
					agent_can_overlap = True
					can_overlap = True
					texture = self.textures[floor].copy()
					if self.LAYOUT["fixed_objects"][y][x].islower():
						item = OBJECTS[ord(self.LAYOUT["fixed_objects"][y][x]) - 97]
						name += f"_{item}"
						agent_can_overlap = False
						can_overlap = (item in SURFACES)
						texture = draw_obj(texture, self.textures[item].copy())
					grid.set_floor(x, y,
							FloorWithObject(name, texture,
								agent_can_overlap=agent_can_overlap,
								can_overlap=can_overlap))



