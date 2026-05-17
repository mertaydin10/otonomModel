"""
train.py - DQL Training Loop
Multi-map agent training with live stats and automatic model saving.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from typing import Optional
from collections import deque

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment
from agent       import DQLAgent


# --- Hyperparameters ---

DEFAULT_CONFIG = {
    # Environment
    "grid_size":       15,
    "random_maps":     True,

    # Training
    "max_episodes":    5_000,
    "max_steps":       200,        # shorter episodes → faster training
    "train_every":     4,

    # Agent
    "learning_rate":   0.001,
    "gamma":           0.99,
    "epsilon_start":   1.0,
    "epsilon_min":     0.01,
    "epsilon_decay":   0.9985,
    "batch_size":      64,
    "buffer_capacity": 30_000,
    "target_update":   10,
    "hidden_size":     128,

    # Saving
    "save_every":      200,
    "model_path":      "models/best_model_v2.pth",
    "stats_path":      "models/training_stats_v2.json",
}

# --- Curriculum Phases (Success Threshold-based) ---
# To advance to next phase: last 100 episodes must meet success threshold.
# min_episodes: minimum episodes in this phase (prevents early advancement).
# Agent must truly learn in a phase, otherwise fails at harder phases.
CURRICULUM = [
    # (obstacle_ratio, min_path, threshold, min_episodes)
    (0.00,  0,  0.75,  300),    # Phase 0: OPEN  - Learn basic navigation, no obstacles
    (0.10,  5,  0.70,  500),    # Phase 1: Light - Add some obstacles
    (0.15,  8,  0.65,  700),    # Phase 2: Medium - Reach 65% success to advance
    (0.22, 10,  0.00, 1000),    # Phase 3: Hard  - Final phase, use remaining time
]


# --- Helper Functions ---

def print_progress(
    episode:       int,
    total:         int,
    reward:        float,
    steps:         int,
    epsilon:       float,
    loss:          Optional[float],
    success_rate:  float,
    window_reward: float,
    elapsed:       float,
) -> None:
    """Terminal progress bar."""
    bar_width  = 20
    progress   = episode / total
    filled     = int(bar_width * progress)
    bar        = "[" + "=" * filled + "-" * (bar_width - filled) + "]"

    loss_str = f"{loss:.4f}" if loss is not None else "  N/A "

    print(
        f"\r[{bar}] {episode:4d}/{total} | "
        f"R:{reward:+7.1f} | "
        f"AvgR:{window_reward:+6.1f} | "
        f"Steps:{steps:3d} | "
        f"eps:{epsilon:.3f} | "
        f"Loss:{loss_str} | "
        f"Succ:{success_rate:.1%} | "
        f"{elapsed:.0f}s",
        end="",
        flush=True,
    )


def train(config: dict, resume: bool = False) -> None:
    """
    Main training loop.

    Args:
        config: Hyperparameter dict
        resume: If True, load existing model and continue training
    """
    # --- Initialize Curriculum Phases ---
    curriculum = config.get("curriculum", CURRICULUM)
    # Current phase index and episode count in current phase
    phase_idx       = 0
    phase_episodes  = 0   # episodes completed in this phase

    # --- Environment and Agent ---
    first_phase = curriculum[0]   # (obs_ratio, min_path, threshold, min_ep)
    env = GridEnvironment(
        size             = config["grid_size"],
        obstacle_ratio   = first_phase[0],
        min_path_length  = first_phase[1],
        max_steps        = config["max_steps"],
        random_maps      = config["random_maps"],
    )

    # v2 training: use 12-element state (no dynamic obstacles)
    # Environment produces 16-element state but last 4 will be zero (dyn=0)
    v2_state_size = 12
    agent = DQLAgent(
        state_size      = env.state_size,
        action_size     = env.action_size,
        hidden_size     = config["hidden_size"],
        learning_rate   = config["learning_rate"],
        gamma           = config["gamma"],
        epsilon_start   = config["epsilon_start"],
        epsilon_min     = config["epsilon_min"],
        epsilon_decay   = config["epsilon_decay"],
        batch_size      = config["batch_size"],
        buffer_capacity = config["buffer_capacity"],
        target_update   = config["target_update"],
    )

    # Devam modunda model yükle
    if resume:
        loaded = agent.load(config["model_path"])
        if loaded:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon)

    print(f"\n{'='*60}")
    print(f" DQL Otonom Surus - Curriculum Egitimi v2")
    print(f"{'='*60}")
    print(f" Grid: {config['grid_size']}x{config['grid_size']}")
    print(f" Max episode: {config['max_episodes']}")
    print(f" Max adim/ep: {config['max_steps']}")
    print(f" Cihaz: {agent.device}")
    print(f"\n Curriculum (basari esikli gecis):")
    for i, (obs, mpl, thr, min_ep) in enumerate(curriculum):
        label = "son faz" if thr == 0 else f"gecis: >={int(thr*100)}% basari"
        print(f"   Faz {i+1}: engel %{int(obs*100):2d} | min yol {mpl:2d} | min {min_ep} ep | {label}")
    print(f"{'='*60}\n")

    # --- Istatistik Takibi ---
    all_rewards:    list = []
    all_successes:  list = []
    all_losses:     list = []
    window_size     = 100
    reward_window   = deque(maxlen=window_size)
    success_window  = deque(maxlen=window_size)
    best_avg_reward = float("-inf")
    start_time      = time.time()

    # --- Episode Loop ---
    for episode in range(1, config["max_episodes"] + 1):

        # --- Curriculum: basari esigi saglandiysa sonraki faza gec ---
        cur_obs, cur_mpl, cur_thr, cur_min = curriculum[phase_idx]
        phase_episodes += 1

        if (phase_idx < len(curriculum) - 1          # not final phase
                and cur_thr > 0                        # threshold defined
                and phase_episodes >= cur_min          # min episodes met
                and len(success_window) == window_size # enough stats
                and float(np.mean(success_window)) >= cur_thr):
            phase_idx      += 1
            phase_episodes  = 0
            cur_obs, cur_mpl, cur_thr, cur_min = curriculum[phase_idx]
            env.obstacle_ratio  = cur_obs
            env.min_path_length = cur_mpl
            print(f"\n  [OK] Curriculum Faz {phase_idx+1}: "
                  f"engel %{int(cur_obs*100)} | "
                  f"min yol {cur_mpl} adim  "
                  f"(basari: {float(np.mean(success_window)):.1%})")

        state        = env.reset()[:v2_state_size]  # 16->12 slice (v2 compat)
        total_reward = 0.0
        ep_losses    = []
        done         = False

        while not done:
            # Select action
            action = agent.select_action(state, training=True)

            # Step
            next_state_full, reward, done, info = env.step(action)
            next_state = next_state_full[:v2_state_size]  # 16->12

            # Remember experience
            agent.remember(state, action, reward, next_state, done)

            # Train every N steps
            if agent.total_steps % config["train_every"] == 0:
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)

            state         = next_state
            total_reward += reward

        # --- End of Episode ---
        reached_goal = info["reached_goal"]
        avg_loss     = float(np.mean(ep_losses)) if ep_losses else None

        # Update stats
        all_rewards.append(total_reward)
        all_successes.append(int(reached_goal))
        all_losses.append(avg_loss)
        reward_window.append(total_reward)
        success_window.append(int(reached_goal))

        # Decay epsilon
        agent.decay_epsilon()

        # Update target network
        if episode % config["target_update"] == 0:
            agent.update_target_network()

        # Show progress
        avg_reward   = float(np.mean(reward_window))
        success_rate = float(np.mean(success_window))
        elapsed      = time.time() - start_time

        print_progress(
            episode       = episode,
            total         = config["max_episodes"],
            reward        = total_reward,
            steps         = info["steps"],
            epsilon       = agent.epsilon,
            loss          = avg_loss,
            success_rate  = success_rate,
            window_reward = avg_reward,
            elapsed       = elapsed,
        )

        # Every 10 episodes print phase info
        if episode % 10 == 0:
            print(f" [Faz {phase_idx+1} | engel%{int(cur_obs*100)} | "
                  f"faz_ep:{phase_episodes}]")

        # Save best model
        if avg_reward > best_avg_reward and len(reward_window) == window_size:
            best_avg_reward = avg_reward
            agent.save(config["model_path"])
            print(f"\n  [BEST] New best model saved (AvgR: {avg_reward:.1f})")

        # Periodic checkpoint
        if episode % config["save_every"] == 0:
            ckpt_path = config["model_path"].replace(".pth", f"_ep{episode}.pth")
            agent.save(ckpt_path)

    # --- Training Complete ---
    print(f"\n\n{'='*60}")
    print(f" [OK] Training complete!")
    print(f" Last 100 episodes success rate: {np.mean(list(success_window)):.1%}")
    print(f" Last 100 episodes avg reward: {np.mean(list(reward_window)):.1f}")
    print(f" Total time: {(time.time()-start_time)/60:.1f} minutes")
    print(f"{'='*60}\n")

    # Save stats
    stats = {
        "config":         config,
        "all_rewards":    all_rewards,
        "all_successes":  all_successes,
        "all_losses":     [l if l is not None else 0.0 for l in all_losses],
        "final_epsilon":  agent.epsilon,
        "total_steps":    agent.total_steps,
        "best_avg_reward": best_avg_reward,
    }
    os.makedirs(os.path.dirname(config["stats_path"]), exist_ok=True)
    with open(config["stats_path"], "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[STATS] Stats saved: {config['stats_path']}")

    # Save final model
    final_path = config["model_path"].replace(".pth", "_final.pth")
    agent.save(final_path)


# --- CLI Interface ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DQL Autonomous Driving Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes",       type=int,   default=DEFAULT_CONFIG["max_episodes"],  help="Number of episodes")
    parser.add_argument("--grid-size",      type=int,   default=DEFAULT_CONFIG["grid_size"],     help="Grid size (NxN)")
    parser.add_argument("--lr",             type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument("--gamma",          type=float, default=DEFAULT_CONFIG["gamma"],         help="Discount factor")
    parser.add_argument("--batch-size",     type=int,   default=DEFAULT_CONFIG["batch_size"],    help="Mini-batch size")
    parser.add_argument("--model-path",     type=str,   default=DEFAULT_CONFIG["model_path"],    help="Model save path")
    parser.add_argument("--resume",         action="store_true",                                 help="Resume from existing model")
    parser.add_argument("--no-random-maps", action="store_true",                                 help="Use fixed map")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config["max_episodes"]  = args.episodes
    config["grid_size"]     = args.grid_size
    config["learning_rate"] = args.lr
    config["gamma"]         = args.gamma
    config["batch_size"]    = args.batch_size
    config["model_path"]    = args.model_path
    config["random_maps"]   = not args.no_random_maps

    # Set working directory to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    train(config, resume=args.resume)
