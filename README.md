![](https://github.com/jlin816/homegrid/raw/main/banner.png)

<p align="center" font-weight="bold">
A minimal home grid world environment to evaluate language understanding in interactive agents.
</p>

# 🏠 Getting Started

Play as a human:
```bash
$ pip install -e .
$ ./homegrid/manual_control.py
```

Use as a gym environment:
```python
import gym
import homegrid
env = gym.make("homegrid-task")
```
See `homegrid/__init__.py` for the environment configurations used in the paper [Learning to Model the World with Language](https://dynalang.github.io/).

# 📑 Documentation

HomeGrid tests whether agents can learn to use language that provides information about the world. In addition to task instructions, the env provides scripted _language hints_, simulating knowledge that agents might learn from humans (e.g., in a collaborative setting) or read in text (e.g., on Wikipedia). Agents navigate around a house to find objects and interact with them to perform tasks, while learning how to understand language from experience.

### ⚡️ Quick Info
- pixel observations (3x3 partial view of the house)
- both one-hot and token embedding observations available
- discrete action space (movement + object interaction)
- 3 rooms, 7 objects (3 trash bins, 4 trash objects)
- multitask with language instructions + hints
- randomized object placement and object dynamics

**Task Templates (38 total tasks):**
- find the `object/bin`: the agent will receive a reward of 1 if it is facing the correct object / bin
- get the `object`: the agent will receive a reward of 1 if it has the correct object in inventory
- put the `object` in the `bin`: the agent will receive a reward of 1 if the bin contains the object
- move the `object` to the `room`: the agent will receive a reward of 1 if the object is in the room
- open the `bin`: the agent will receive a reward of 1 if the bin is in the open state

**Language Types and Templates**

- **Future Observations**: descriptions of what agents might observe in the future, such as "The plates are in the kitchen."
    - _"`object/bin` is in the `room`"_: the object or bin is in the indicated room
    - _"i moved the `object` to the `room`"_: the object has been moved to the room
    - _"there will be `object` in the `room`"_: the object will spawn in the room in five timesteps
- **Dynamics**: descriptions of environment dynamics, such as "Pedal to open the compost bin."
    - _"`action` to open the `bin`"_: the indicated action is the correct action to open the bin
- **Corrections**: interactive, task-specific feedback based on what the agent is currently doing, such as "Turn around."
    - _"no, turn around"_: the agent's distance to the current goal object or bin (given the task) has increased compared to the last timestep

Environment instances are provided for task instruction + each of the types above in `homegrid/__init__.py`.

Language is provided by `homegrid/language_wrappers.py` and streamed one token per timestep by default. Both token IDs and token embeddings are provided in the observation, using the [T5](https://arxiv.org/abs/1910.10683) tokenizer and encoder model. The original paper introducing HomeGrid uses the token IDs.
Some strings are higher priority than others and may interrupt a string that is currently being read. By default, the environment will stream some hints that apply to a whole episode during the first timesteps, while the agent does not move. See `homegrid/language_wrappers.py` for details.

### Observation Space

For the full HomeGrid environment with language:

- `image (uint8 (96, 96, 3))`: pixel agent-centric local view
- `token (int)`: T5 token ID of the token at the current timestep
- `token_embed (float32 (512,))`: T5 embedding of the token at the current timestep
- `is_read_step (bool)`: for logging, `True` if agent is reading strings before the episode begins
- `log_language_info (str)`: for logging, human-readable text for the string currently being streamed

# 💻 Development

New development and extensions to the environment are welcome!

### Adding new language utterances

Sentences are pre-embedded and cached into a file for training efficiency. You'll have to append the additional sentences to `homegrid/homegrid_sentences.txt` and re-generate the cached token and embedding file with the following command:
```bash
python scripts/embed_offline.py \
    --infile homegrid/homegrid_sentences.txt \
    --outfile homegrid/homecook_embeds.pkl \
    --model t5
```

### Adding new layouts and objects

HomeGrid currently has one layout and a fixed set of objects that are sampled to populate each episode. Many of the receptacles and containers (e.g. cabinets) are disabled for simplicity.

To add new layouts, create a new class in `homegrid/layout.py`.

To add new static (non-interactive) objects, add assets to `homegrid/assets.py` and then specify where they are rendered in the `homegrid/layout.py`.

To add new interactive objects, additionally specify how they behave in `homegrid/homegrid_base.py:step`.

# Acknowledgments

HomeGrid is based on [MiniGrid](https://github.com/Farama-Foundation/Minigrid).
The environment assets are thanks to [limezu](https://limezu.itch.io/) and [Mounir Tohami](https://mounirtohami.itch.io/).

# Citation

```
@article{lin2023learning,
         title={Learning to Model the World with Language},
         author={Jessy Lin and Yuqing Du and Olivia Watkins and Danijar Hafner and Pieter Abbeel and Dan Klein and Anca Dragan},
         year={2023},
         eprint={2308.01399},
         archivePrefix={arXiv},
}
```
