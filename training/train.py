"""
train.py — DQL Eğitim Döngüsü
Çoklu harita üzerinde ajan eğitimi, canlı istatistik çıktısı
ve otomatik model kaydetme içerir.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from typing import Optional
from collections import deque

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment
from agent       import DQLAgent


# ─── Hiperparametreler ────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    # Ortam
    "grid_size":       15,
    "obstacle_ratio":  0.15,
    "random_maps":     True,

    # Eğitim
    "max_episodes":    5000,    # 2000 → 5000: daha fazla harita görerek genelleme artar
    "max_steps":       500,
    "train_every":     4,       # Her N adımda bir güncelle

    # Ajan
    "learning_rate":   0.001,
    "gamma":           0.99,    # 0.95 → 0.99: uzun vadeli planlama iyileşir
    "epsilon_start":   1.0,
    "epsilon_min":     0.01,
    "epsilon_decay":   0.998,   # 0.995 → 0.998: keşif süresi uzuyor (~2300 ep'e kadar)
    "batch_size":      64,
    "buffer_capacity": 20_000,  # 10_000 → 20_000: daha çeşitli deneyim
    "target_update":   10,      # Episode bazında
    "hidden_size":     128,     # 64 → 128: daha güçlü ağ

    # Kayıt
    "save_every":      200,     # Her N episode'da kaydet
    "model_path":      "models/best_model.pth",
    "stats_path":      "models/training_stats.json",
}


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

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
    """Terminal ilerleme çubuğu."""
    bar_width  = 20
    progress   = episode / total
    filled     = int(bar_width * progress)
    bar        = "█" * filled + "░" * (bar_width - filled)

    loss_str = f"{loss:.4f}" if loss is not None else "  N/A "

    print(
        f"\r[{bar}] {episode:4d}/{total} | "
        f"R:{reward:+7.1f} | "
        f"AvgR:{window_reward:+6.1f} | "
        f"Steps:{steps:3d} | "
        f"ε:{epsilon:.3f} | "
        f"Loss:{loss_str} | "
        f"Succ:{success_rate:.1%} | "
        f"{elapsed:.0f}s",
        end="",
        flush=True,
    )


def train(config: dict, resume: bool = False) -> None:
    """
    Ana eğitim döngüsü.

    Args:
        config: Hiperparametre sözlüğü
        resume: True ise mevcut modeli yükleyerek devam et
    """
    # ── Ortam ve Ajan ──────────────────────────────────────────────────────
    env = GridEnvironment(
        size           = config["grid_size"],
        obstacle_ratio = config["obstacle_ratio"],
        max_steps      = config["max_steps"],
        random_maps    = config["random_maps"],
    )

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
    print(f" 🚗 DQL Otonom Sürüş Eğitimi")
    print(f"{'='*60}")
    print(f" Grid: {config['grid_size']}×{config['grid_size']}")
    print(f" Engel Oranı: {config['obstacle_ratio']:.0%}")
    print(f" Episode: {config['max_episodes']}")
    print(f" Cihaz: {agent.device}")
    print(f"{'='*60}\n")

    # ── İstatistik Takibi ──────────────────────────────────────────────────
    all_rewards:    list = []
    all_successes:  list = []
    all_losses:     list = []
    window_size     = 100
    reward_window   = deque(maxlen=window_size)
    success_window  = deque(maxlen=window_size)
    best_avg_reward = float("-inf")
    start_time      = time.time()

    # ── Episode Döngüsü ────────────────────────────────────────────────────
    for episode in range(1, config["max_episodes"] + 1):
        state        = env.reset()
        total_reward = 0.0
        ep_losses    = []
        done         = False

        while not done:
            # Aksiyon seç
            action = agent.select_action(state, training=True)

            # Adımı at
            next_state, reward, done, info = env.step(action)

            # Deneyimi kaydet
            agent.remember(state, action, reward, next_state, done)

            # Her N adımda eğit
            if agent.total_steps % config["train_every"] == 0:
                loss = agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)

            state         = next_state
            total_reward += reward

        # ── Episode Sonu ───────────────────────────────────────────────────
        reached_goal = info["reached_goal"]
        avg_loss     = float(np.mean(ep_losses)) if ep_losses else None

        # İstatistikleri güncelle
        all_rewards.append(total_reward)
        all_successes.append(int(reached_goal))
        all_losses.append(avg_loss)
        reward_window.append(total_reward)
        success_window.append(int(reached_goal))

        # Epsilon azalt
        agent.decay_epsilon()

        # Hedef ağı güncelle
        if episode % config["target_update"] == 0:
            agent.update_target_network()

        # İlerlemeyi göster
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

        # Her 10 episode'da yeni satır
        if episode % 10 == 0:
            print()

        # En iyi modeli kaydet
        if avg_reward > best_avg_reward and len(reward_window) == window_size:
            best_avg_reward = avg_reward
            agent.save(config["model_path"])
            print(f"\n  💾 Yeni en iyi model kaydedildi (AvgR: {avg_reward:.1f})")

        # Periyodik checkpoint
        if episode % config["save_every"] == 0:
            ckpt_path = config["model_path"].replace(".pth", f"_ep{episode}.pth")
            agent.save(ckpt_path)

    # ── Eğitim Sonu ──────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print(f" ✅ Eğitim tamamlandı!")
    print(f" Son 100 episode başarı oranı: {np.mean(list(success_window)):.1%}")
    print(f" Son 100 episode ortalama ödül: {np.mean(list(reward_window)):.1f}")
    print(f" Toplam süre: {(time.time()-start_time)/60:.1f} dakika")
    print(f"{'='*60}\n")

    # İstatistikleri kaydet
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
    print(f"📊 İstatistikler kaydedildi: {config['stats_path']}")

    # Son model kaydı
    final_path = config["model_path"].replace(".pth", "_final.pth")
    agent.save(final_path)


# ─── CLI Arayüzü ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🚗 DQL Otonom Sürüş Eğitimi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes",       type=int,   default=DEFAULT_CONFIG["max_episodes"],   help="Episode sayısı")
    parser.add_argument("--grid-size",      type=int,   default=DEFAULT_CONFIG["grid_size"],      help="Grid boyutu (NxN)")
    parser.add_argument("--obstacle-ratio", type=float, default=DEFAULT_CONFIG["obstacle_ratio"], help="Engel yoğunluğu")
    parser.add_argument("--lr",             type=float, default=DEFAULT_CONFIG["learning_rate"],  help="Öğrenme hızı")
    parser.add_argument("--gamma",          type=float, default=DEFAULT_CONFIG["gamma"],          help="İndirim faktörü")
    parser.add_argument("--batch-size",     type=int,   default=DEFAULT_CONFIG["batch_size"],     help="Mini-batch boyutu")
    parser.add_argument("--model-path",     type=str,   default=DEFAULT_CONFIG["model_path"],     help="Model kayıt yolu")
    parser.add_argument("--resume",         action="store_true",                                  help="Mevcut modelden devam et")
    parser.add_argument("--no-random-maps", action="store_true",                                  help="Sabit harita kullan")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config["max_episodes"]   = args.episodes
    config["grid_size"]      = args.grid_size
    config["obstacle_ratio"] = args.obstacle_ratio
    config["learning_rate"]  = args.lr
    config["gamma"]          = args.gamma
    config["batch_size"]     = args.batch_size
    config["model_path"]     = args.model_path
    config["random_maps"]    = not args.no_random_maps

    # Çalışma dizinini proje köküne ayarla
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    train(config, resume=args.resume)
