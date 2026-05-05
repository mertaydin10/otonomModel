"""
tools/plot_stats.py — Eğitim istatistiklerini görselleştir

Kullanım:
  python tools/plot_stats.py
  python tools/plot_stats.py --stats models/training_stats.json
  python tools/plot_stats.py --text        # Grafik yerine terminal çıktısı
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_text_summary(stats: dict) -> None:
    """Grafik kütüphanesi olmadan terminal çıktısı."""
    rewards   = stats["all_rewards"]
    successes = stats["all_successes"]
    losses    = stats["all_losses"]
    n         = len(rewards)
    window    = 100

    print(f"\n{'='*60}")
    print(f"  Eğitim İstatistikleri — {n} Episode")
    print(f"{'='*60}")

    # Her 100 episode özet
    print(f"\n{'Ep Aralığı':<15} {'AvgReward':>10} {'Başarı%':>9} {'AvgLoss':>9}")
    print("-" * 47)

    for start in range(0, n, window):
        end   = min(start + window, n)
        chunk_r = rewards[start:end]
        chunk_s = successes[start:end]
        chunk_l = [l for l in losses[start:end] if l and l > 0]

        avg_r    = sum(chunk_r) / len(chunk_r)
        avg_s    = sum(chunk_s) / len(chunk_s) * 100
        avg_l    = sum(chunk_l) / len(chunk_l) if chunk_l else 0.0

        bar_len  = int(avg_s / 5)
        bar      = "█" * bar_len + "░" * (20 - bar_len)

        print(f"{start+1:>5}-{end:<8}  {avg_r:>+9.1f}  {avg_s:>7.1f}%  {avg_l:>8.4f}  |{bar}|")

    # Genel özet
    total_s = sum(successes)
    best_ep = rewards.index(max(rewards)) + 1
    print(f"\n{'='*60}")
    print(f"  Toplam başarı       : {total_s}/{n} (%{total_s/n*100:.1f})")
    print(f"  En yüksek tek ödül  : {max(rewards):+.1f}  (episode {best_ep})")
    print(f"  En düşük tek ödül   : {min(rewards):+.1f}")
    print(f"  Son 100 ep başarı   : %{sum(successes[-100:])/100*100:.1f}")
    print(f"  Son 100 ep ort ödül : {sum(rewards[-100:])/100:+.1f}")
    print(f"  Final epsilon       : {stats['final_epsilon']:.4f}")
    print(f"  Toplam adım         : {stats['total_steps']:,}")
    print(f"  En iyi avg reward   : {stats['best_avg_reward']:+.1f}")
    print(f"{'='*60}\n")


def plot_graphs(stats: dict) -> None:
    """matplotlib ile 4 grafik çiz."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("[!] matplotlib bulunamadı: pip install matplotlib")
        print("    Bunun yerine --text modu kullanılıyor...\n")
        print_text_summary(stats)
        return

    rewards   = np.array(stats["all_rewards"])
    successes = np.array(stats["all_successes"])
    losses    = np.array([l if l else 0.0 for l in stats["all_losses"]])
    n         = len(rewards)
    eps       = np.arange(1, n + 1)

    def moving_avg(arr, w=100):
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("DQL Otonom Sürüş — Eğitim İstatistikleri", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#1e1e2e")

    colors = {"reward": "#89b4fa", "avg": "#a6e3a1", "success": "#fab387",
              "loss": "#f38ba8", "epsilon": "#cba6f7", "grid": "#313244"}

    for ax in axes.flat:
        ax.set_facecolor("#181825")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor(colors["grid"])

    # ── 1. Episode Ödülleri ───────────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(eps, rewards, color=colors["reward"], alpha=0.25, linewidth=0.6, label="Episode ödülü")
    if n >= 100:
        ma = moving_avg(rewards)
        ax1.plot(eps[99:], ma, color=colors["avg"], linewidth=2, label="100 ep ortalama")
    ax1.axhline(0, color="#585b70", linewidth=0.8, linestyle="--")
    ax1.set_title("Episode Ödülleri")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Toplam Ödül")
    ax1.legend(facecolor="#313244", labelcolor="white", fontsize=8)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%+.0f"))

    # ── 2. Başarı Oranı ───────────────────────────────────────────────────────
    ax2 = axes[0, 1]
    if n >= 100:
        success_rate = moving_avg(successes.astype(float)) * 100
        ax2.plot(eps[99:], success_rate, color=colors["success"], linewidth=2)
        ax2.fill_between(eps[99:], success_rate, alpha=0.2, color=colors["success"])
    ax2.axhline(80, color="#585b70", linewidth=0.8, linestyle="--", label="%80 hedef")
    ax2.set_title("Başarı Oranı (100 ep kayan ortalama)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Başarı (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(facecolor="#313244", labelcolor="white", fontsize=8)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

    # ── 3. Kayıp (Loss) ───────────────────────────────────────────────────────
    ax3 = axes[1, 0]
    nonzero_mask = losses > 0
    if nonzero_mask.sum() > 10:
        ax3.scatter(eps[nonzero_mask], losses[nonzero_mask],
                    color=colors["loss"], alpha=0.3, s=3, label="Adım kaybı")
        if nonzero_mask.sum() >= 100:
            loss_ma = moving_avg(losses[nonzero_mask])
            loss_eps = eps[nonzero_mask][99:]
            ax3.plot(loss_eps, loss_ma, color="#ff8c8c", linewidth=2, label="100 ep ortalama")
    ax3.set_title("Eğitim Kaybı (Loss)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Huber Loss")
    ax3.legend(facecolor="#313244", labelcolor="white", fontsize=8)
    ax3.set_yscale("log")

    # ── 4. Epsilon Azalması ──────────────────────────────────────────────────
    ax4 = axes[1, 1]
    epsilon_start = stats["config"]["epsilon_start"]
    epsilon_min   = stats["config"]["epsilon_min"]
    epsilon_decay = stats["config"]["epsilon_decay"]
    epsilon_curve = [max(epsilon_min, epsilon_start * (epsilon_decay ** i)) for i in range(n)]
    ax4.plot(eps, epsilon_curve, color=colors["epsilon"], linewidth=2)
    ax4.fill_between(eps, epsilon_curve, epsilon_min, alpha=0.15, color=colors["epsilon"])
    ax4.axhline(epsilon_min, color="#585b70", linewidth=0.8, linestyle="--",
                label=f"Min ε = {epsilon_min}")
    ax4.set_title("Epsilon Azalması (Keşif → Sömürü)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon (ε)")
    ax4.set_ylim(-0.02, 1.05)
    ax4.legend(facecolor="#313244", labelcolor="white", fontsize=8)

    plt.tight_layout()
    out = "models/training_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  ✅ Grafik kaydedildi: {out}")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Eğitim istatistiklerini görselleştir")
    parser.add_argument("--stats", default="models/training_stats.json",
                        help="İstatistik JSON dosyası")
    parser.add_argument("--text", action="store_true",
                        help="Grafik yerine terminal çıktısı")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    if not os.path.exists(args.stats):
        print(f"[HATA] Dosya bulunamadı: {args.stats}")
        print("  Önce eğitimi çalıştır: python training/train.py")
        sys.exit(1)

    with open(args.stats, encoding="utf-8") as f:
        stats = json.load(f)

    if args.text:
        print_text_summary(stats)
    else:
        plot_graphs(stats)
        print_text_summary(stats)
