#!/usr/bin/env python3

import gym

from homegrid.window import Window
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("t5-small")

def redraw(window, img):
    window.show_img(img)


def reset(env, window, seed=None, agent_view=False):
    obs, _ = env.reset()
    img = obs["image"] if agent_view else env.get_frame()
    redraw(window, img)


def step(env, window, action, agent_view=False):
    obs, reward, terminated, truncated, info = env.step(action)
    print(info["symbolic_state"])
    token = tok.decode([obs["token"]])
    print(f"step={env.step_cnt}, reward={reward:.2f}")
    print("Token: ", token)
    print("Language: ", obs["log_language_info"] if "log_language_info" in obs else "None")
    print("Task: ", env.task)
    print("-"*20)
    window.set_caption(
        f"r={reward:.2f} token_id={obs['token']} token="
        f"{token} \ncurrent: {obs['log_language_info'][:50]}...")

    if terminated:
        print(f"terminated! r={reward}")
        reset(env, window)
    elif truncated:
        print("truncated!")
        reset(env, window)
    else:
        img = obs["image"] if agent_view else env.get_frame()
        redraw(window, img)


def key_handler(env, window, event, agent_view=False):
    print("pressed", event.key)
    step_ = lambda a: step(env, window, a, agent_view)

    if event.key == "escape":
        window.close()
        return

    if event.key == "backspace":
        reset(env, window)
        return

    if event.key == "left":
        step_(env.actions.left)
        return
    if event.key == "right":
        step_(env.actions.right)
        return
    if event.key == "up":
        step_(env.actions.up)
        return
    if event.key == "down":
        step_(env.actions.down)
        return

    if event.key == "k":
        step_(env.actions.pickup)
        return
    if event.key == "d":
        step_(env.actions.drop)
        return
    if event.key == "g":
        step_(env.actions.get)
        return
    if event.key == "p":
        step_(env.actions.pedal)
        return
    if event.key == "r":
        step_(env.actions.grasp)
        return
    if event.key == "l":
        step_(env.actions.lift)
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", help="gym environment to load", default="homegrid-task"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()
    env = gym.make(args.env, disable_env_checker=True)

    for k in plt.rcParams:
      if "keymap" in k:
        plt.rcParams[k] = []
    window = Window("homegrid - " + args.env)

    window.reg_key_handler(lambda event: key_handler(env, window, event,
                                                     args.agent_view))

    seed = None if args.seed == -1 else args.seed
    reset(env, window, seed, args.agent_view)

    # Blocking event loop
    window.show(block=True)
