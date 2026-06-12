"""
This script demonstrates how to evaluate a pretrained smolVLA policy on the LIBERO benchmark.
https://github.com/huggingface/lerobot/issues/1316
"""

import collections
import dataclasses
import logging
import math
import pathlib
import os, json, re
from typing import Any
from pathlib import Path

import cv2
import draccus
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def build_suite_category(suite):
    from libero.libero.benchmark import libero_suite_task_map
    task_cls = os.path.join(os.path.dirname(libero_suite_task_map.__file__), "task_classification.json")
    with open(task_cls, "r", encoding="utf-8") as f: data = json.load(f)
    cat2ids = {}
    for it in data[suite]:
        cat2ids[it["id"]-1] = it["category"]
    return cat2ids

def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    """
    Load init states for task `i` with the same path heuristics as LIBERO-plus.

    Newer LIBERO datasets rename init_state files (e.g. strip `_table_XX`, `_tb_XX`,
    `_light_`, `_add_`, `_level` prefixes). The upstream Benchmark.get_task_init_states
    already implements this logic; we reuse it when available, and fall back to the
    same heuristics here for compatibility.
    """
    # Prefer the task_suite's own implementation (includes all path tweaks).
    if hasattr(task_suite, "get_task_init_states"):
        try:
            return task_suite.get_task_init_states(i)
        except Exception:
            # Fall back to local heuristic below
            pass

    task = task_suite.tasks[i]
    init_states_dir = Path(get_libero_path("init_states"))
    candidates: list[Path] = []

    # Original path
    candidates.append(init_states_dir / task.problem_folder / task.init_states_file)

    # Heuristics following LIBERO-plus benchmark/__init__.py
    fname = task.init_states_file

    if "_language_" in fname:
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            task.problem_folder,
            fname.split("_language_")[0] + "." + fname.split(".")[-1],
        )
        candidates.append(Path(init_states_path))
    else:
        if "_view_" in fname:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                task.problem_folder,
                fname.split("_view_")[0] + "." + fname.split(".")[-1],
            )
            candidates.append(Path(init_states_path))
        else:
            if "_table_" in fname:
                init_states_path = os.path.join(
                    get_libero_path("init_states"),
                    task.problem_folder,
                    re.sub(r'_table_\d+', '', fname),
                )
                candidates.append(Path(init_states_path))
            if "_tb_" in fname:
                init_states_path = os.path.join(
                    get_libero_path("init_states"),
                    task.problem_folder,
                    re.sub(r'_tb_\d+', '', fname),
                )
                candidates.append(Path(init_states_path))

            if "_light_" in fname:
                init_states_path = os.path.join(
                    get_libero_path("init_states"),
                    task.problem_folder,
                    fname.split("_light_")[0] + "." + fname.split(".")[-1],
                )
                candidates.append(Path(init_states_path))

            if "_add_" in fname or "_level" in fname:
                init_states_path = os.path.join(
                    get_libero_path("init_states"),
                    "libero_newobj",
                    task.problem_folder,
                    fname,
                )
                candidates.append(Path(init_states_path))

    for path in candidates:
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)  # nosec B614

    raise FileNotFoundError(f"Init states not found for task: {task.name}; tried: {candidates}")

@dataclasses.dataclass
class Args:
    """
    Evaluation arguments for smolVLA on LIBERO.
    """

    # --- Hugging Face arguments ---
    policy_path: str = "lerobot/smolvla_base"
    """Path to the pretrained policy on the Hugging Face Hub or local directory."""

    # --- LIBERO environment-specific parameters ---
    task_suite_name: str = "libero_spatial"
    """Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    num_steps_wait: int = 10
    """Number of steps to wait for objects to stabilize in sim."""
    num_trials_per_task: int = 1
    """Number of rollouts per task."""

    # --- Evaluation arguments ---
    video_out_path: str = "data/libero/videos"
    """Path to save videos."""
    device: str = "cuda"
    """Device to use for evaluation."""

    seed: int = 0
    """Random Seed (for reproducibility)"""


@draccus.wrap()
def eval_libero(args: Args) -> None:
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- Load Policy ---
    logging.info(f"policy_path: {args.policy_path}")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(args.device)
    policy.eval()

    # --- Initialize LIBERO task suite ---
    benchmark_dict = benchmark.get_benchmark_dict()
    try:
        task_suite = benchmark_dict[args.task_suite_name]()
    except KeyError:
        raise ValueError(
            f"Unknown task suite: {args.task_suite_name}. "
            f"Available options are: {list(benchmark_dict.keys())}"
        )
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        # Fallback for custom task suites
        max_steps = 520
    
    task_id2cat = build_suite_category(args.task_suite_name)
    eval_info=collections.defaultdict(list)

    # --- Evaluation Loop ---
    total_episodes, total_successes = 0, 0
    for task_id in tqdm(range(num_tasks_in_suite), desc="Tasks"):
        # Get task
        # import pdb; pdb.set_trace()
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = get_task_init_states(task_suite, task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm(
            range(args.num_trials_per_task),
            desc=f"Task {task_id}: {task.language}",
            leave=False,
        ):
            logging.info(f"\nTask: {task_description}")

            # Reset environment and policy
            env.reset()
            policy.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            for _ in range(args.num_steps_wait):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

            # Setup
            t = 0
            frames = []
            done = False

            # Add initial frame
            agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            # frames.append(agentview_image)
            # import ipdb; ipdb.set_trace()
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:
                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    agentview_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    frames.append(agentview_image)

                    # Prepare observations dict
                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )
                    observation = {
                        "observation.images.front": torch.from_numpy(agentview_image / 255.0)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(args.device).unsqueeze(0),
                        "observation.images.wrist": torch.from_numpy(wrist_img / 255.0)
                        .permute(2, 0, 1)
                        .to(torch.float32)
                        .to(args.device).unsqueeze(0),
                        "observation.state": torch.from_numpy(state).to(torch.float32).to(args.device).unsqueeze(0),
                        "task": task_description,
                    }

                    # Query model to get action
                    with torch.inference_mode():
                        action_tensor = policy.select_action(observation)
                    action = action_tensor.cpu().numpy()[0]
                    action = action[:7]

                    # Execute action in environment
                    obs, _, done, info = env.step(action)
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1
            eval_info[task_id2cat[task_id]].append(done)

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_").replace("/", "_")
            video_path = (
                pathlib.Path(args.video_out_path) / f"rollout_task_{task_id}_episode_{episode_idx}_{suffix}.mp4" #_{task_segment}
            )
            fps = 30
            writer = imageio.get_writer(video_path, fps=fps)

            for image in frames:
                writer.append_data(image)
            writer.close()
            logging.info(f"Saved video to {video_path}")

            # Log current results
            logging.info(f"Success: {done}")
            if total_episodes > 0:
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.4f}%)")

        # Log final results for the task
        if task_episodes > 0:
            logging.info(f"Task {task_id} success rate: {float(task_successes) / float(task_episodes):.4f}")
        if total_episodes > 0:
            logging.info(f"Cumulative success rate: {float(total_successes) / float(total_episodes):.4f}")

    logging.info("--- Evaluation finished ---")
    if total_episodes > 0:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes):.4f}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Total successes: {total_successes}")

    # Save info
    done_state=[]
    for task_name in eval_info:
        done_state.extend(eval_info[task_name])
        logging.info(f"Success rate of {task_name}: {np.mean(eval_info[task_name]).item():.4f}")
    logging.info(f"Total success rate: {np.mean(done_state).item():.4f}")
    with open(Path(args.video_out_path) / "eval_info.json", "w") as f:
        json.dump(eval_info, f, indent=2)


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    # task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    task_description = getattr(env, "language_instruction", task.language)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("evaluation_log.txt"),
            logging.StreamHandler()  # Optional: keeps logging in the terminal too
        ]
    )
    eval_libero()