"""
train_v3.py — DQL Eğitim Döngüsü (v3 — Hareketli Engelli)

Statik engellerden başlayıp kademeli olarak hareketli engeller eklenen
curriculum eğitimi. Mevcut v2 modelinden transfer learning destekler.
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
    "random_maps":     True,

    # Eğitim
    "max_episodes":    15_000,
    "max_steps":       400,         # dinamik engelli haritalarda daha fazla süre
    "train_every":     4,

    # Ajan
    "learning_rate":   0.0005,      # daha düşük lr — transfer learning için stabil
    "gamma":           0.99,
    "epsilon_start":   1.0,
    "epsilon_min":     0.01,
    "epsilon_decay":   0.9990,      # daha yavaş decay — keşif daha uzun sürer
    "batch_size":      64,
    "buffer_capacity": 50_000,      # daha büyük buffer — çeşitli deneyimler
    "target_update":   10,
    "hidden_size":     256,         # büyük ağ — dinamik ortam daha karmaşık

    # Kayıt
    "save_every":      300,
    "model_path":      "models/best_model_v3.pth",
    "stats_path":      "models/training_stats_v3.json",
}

# ─── Curriculum Aşamaları (Yoğun Hareketli Engel) ────────────────────────────
# (engel_oranı, min_yol, dyn_engel_sayısı, dyn_hareket_interval,
#  geçiş_başarı_eşiği, min_episode)
#
# Hareketli engel arttıkça statik engel azalır — ajan çaresiz kalmasın.
# Toplam zorluk dengeli: az statik + çok dinamik = zor ama adil.

CURRICULUM = [
    # (obs_ratio, min_path, dyn_count, dyn_interval, threshold, min_ep)
    (0.15,  5, 3, 1, 0.55, 1500),    # Faz 1: 3 dyn, %15 statik — ısınma
    (0.13,  5, 5, 1, 0.50, 2000),    # Faz 2: 5 dyn, %13 statik
    (0.10,  5, 6, 1, 0.45, 2500),    # Faz 3: 6 dyn, %10 statik
    (0.08,  5, 7, 1, 0.40, 3000),    # Faz 4: 7 dyn, %8 statik
    (0.08,  5, 8, 1, 0.00, 4000),    # Faz 5: Son faz — 8 dyn, %8 statik
]


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
    phase:         int,
    dyn_count:     int,
) -> None:
    """Terminal ilerleme çubuğu."""
    bar_width  = 20
    progress   = episode / total
    filled     = int(bar_width * progress)
    bar        = "█" * filled + "░" * (bar_width - filled)

    loss_str = f"{loss:.4f}" if loss is not None else "  N/A "

    dyn_str = f"Dyn:{dyn_count}" if dyn_count > 0 else "Static"

    print(
        f"\r[{bar}] {episode:4d}/{total} | "
        f"R:{reward:+7.1f} | "
        f"AvgR:{window_reward:+6.1f} | "
        f"Steps:{steps:3d} | "
        f"ε:{epsilon:.3f} | "
        f"Loss:{loss_str} | "
        f"Succ:{success_rate:.1%} | "
        f"{dyn_str} | "
        f"F{phase} | "
        f"{elapsed:.0f}s",
        end="",
        flush=True,
    )


def train(config: dict, resume: bool = False) -> None:
    """
    Ana eğitim döngüsü (v3 — hareketli engelli).

    Args:
        config: Hiperparametre sözlüğü
        resume: True ise mevcut modeli yükleyerek devam et
    """
    # ── Curriculum aşamalarını belirle ────────────────────────────────────
    curriculum = config.get("curriculum", CURRICULUM)
    phase_idx       = 0
    phase_episodes  = 0

    # ── Ortam ve Ajan ──────────────────────────────────────────────────────
    first_phase = curriculum[0]
    env = GridEnvironment(
        size                   = config["grid_size"],
        obstacle_ratio         = first_phase[0],
        min_path_length        = first_phase[1],
        max_steps              = config["max_steps"],
        random_maps            = config["random_maps"],
        dynamic_obstacle_count = first_phase[2],
        dynamic_move_interval  = first_phase[3],
    )

    agent = DQLAgent(
        state_size      = env.state_size,    # 16
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

    print(f"\n{'='*65}")
    print(f" DQL Otonom Sürüş — Hareketli Engel Eğitimi v3")
    print(f"{'='*65}")
    print(f" Grid: {config['grid_size']}×{config['grid_size']}")
    print(f" State boyutu: {env.state_size}")
    print(f" Hidden size: {config['hidden_size']}")
    print(f" Max episode: {config['max_episodes']}")
    print(f" Max adım/ep: {config['max_steps']}")
    print(f" Cihaz: {agent.device}")
    print(f"\n Curriculum (hareketli engelli):")
    for i, (obs, mpl, dyn, dint, thr, min_ep) in enumerate(curriculum):
        label = "son faz" if thr == 0 else f"geçiş: ≥%{int(thr*100)} başarı"
        dyn_label = f"{dyn} dyn (int:{dint})" if dyn > 0 else "statik"
        print(f"   Faz {i+1}: engel %{int(obs*100):2d} | min yol {mpl:2d} | "
              f"{dyn_label:16s} | min {min_ep} ep | {label}")
    print(f"{'='*65}\n")

    # ── İstatistik Takibi ──────────────────────────────────────────────────
    all_rewards:    list = []
    all_successes:  list = []
    all_losses:     list = []
    window_size     = 100
    reward_window   = deque(maxlen=window_size)
    success_window  = deque(maxlen=window_size)
    best_avg_reward = float("-inf")
    start_time      = time.time()

    # Mevcut faz değerleri
    cur_dyn_count = first_phase[2]

    # ── Episode Döngüsü ────────────────────────────────────────────────────
    for episode in range(1, config["max_episodes"] + 1):

        # ── Curriculum: başarı eşiği sağlandıysa sonraki faza geç ──────────
        cur_obs, cur_mpl, cur_dyn, cur_dint, cur_thr, cur_min = curriculum[phase_idx]
        phase_episodes += 1

        if (phase_idx < len(curriculum) - 1          # son faz değilse
                and cur_thr > 0                        # eşik tanımlıysa
                and phase_episodes >= cur_min          # minimum episode dolmuşsa
                and len(success_window) == window_size # yeterli istatistik varsa
                and float(np.mean(success_window)) >= cur_thr):
            phase_idx      += 1
            phase_episodes  = 0
            cur_obs, cur_mpl, cur_dyn, cur_dint, cur_thr, cur_min = curriculum[phase_idx]
            env.obstacle_ratio  = cur_obs
            env.min_path_length = cur_mpl
            env.dynamic_obstacle_count = cur_dyn
            env.dynamic_move_interval  = cur_dint
            cur_dyn_count = cur_dyn
            dyn_label = f"{cur_dyn} dyn engel (int:{cur_dint})" if cur_dyn > 0 else "statik"
            print(f"\n  ✅ Curriculum Faz {phase_idx+1}: "
                  f"engel %{int(cur_obs*100)} | "
                  f"min yol {cur_mpl} adım | "
                  f"{dyn_label}  "
                  f"(başarı: {float(np.mean(success_window)):.1%})")

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
            phase         = phase_idx + 1,
            dyn_count     = cur_dyn_count,
        )

        # Her 10 episode'da faz bilgisiyle yeni satır
        if episode % 10 == 0:
            dyn_info = f"dyn:{cur_dyn_count}" if cur_dyn_count > 0 else "static"
            print(f" [Faz {phase_idx+1} | engel%{int(cur_obs*100)} | "
                  f"{dyn_info} | faz_ep:{phase_episodes}]")

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
    print(f"\n\n{'='*65}")
    print(f" ✅ Eğitim tamamlandı!")
    print(f" Son 100 episode başarı oranı: {np.mean(list(success_window)):.1%}")
    print(f" Son 100 episode ortalama ödül: {np.mean(list(reward_window)):.1f}")
    print(f" Toplam süre: {(time.time()-start_time)/60:.1f} dakika")
    print(f"{'='*65}\n")

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
        description="🚗 DQL Otonom Sürüş — Hareketli Engel Eğitimi v3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes",       type=int,   default=DEFAULT_CONFIG["max_episodes"],  help="Episode sayısı")
    parser.add_argument("--grid-size",      type=int,   default=DEFAULT_CONFIG["grid_size"],     help="Grid boyutu (NxN)")
    parser.add_argument("--lr",             type=float, default=DEFAULT_CONFIG["learning_rate"], help="Öğrenme hızı")
    parser.add_argument("--gamma",          type=float, default=DEFAULT_CONFIG["gamma"],         help="İndirim faktörü")
    parser.add_argument("--batch-size",     type=int,   default=DEFAULT_CONFIG["batch_size"],    help="Mini-batch boyutu")
    parser.add_argument("--hidden-size",    type=int,   default=DEFAULT_CONFIG["hidden_size"],   help="Hidden katman boyutu")
    parser.add_argument("--model-path",     type=str,   default=DEFAULT_CONFIG["model_path"],    help="Model kayıt yolu")
    parser.add_argument("--resume",         action="store_true",                                 help="Mevcut modelden devam et")
    parser.add_argument("--no-random-maps", action="store_true",                                 help="Sabit harita kullan")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config["max_episodes"]  = args.episodes
    config["grid_size"]     = args.grid_size
    config["learning_rate"] = args.lr
    config["gamma"]         = args.gamma
    config["batch_size"]    = args.batch_size
    config["hidden_size"]   = args.hidden_size
    config["model_path"]    = args.model_path
    config["random_maps"]   = not args.no_random_maps

    # Çalışma dizinini proje köküne ayarla
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    train(config, resume=args.resume)
