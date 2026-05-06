"""
tools/watch.py — Ajanın haritada hareketlerini canlı izle

Kullanım:
  python tools/watch.py                            # Varsayılan: best_model.pth, no-revisit aktif
  python tools/watch.py --epsilon 0.05             # Ek küçük rastgelelik
  python tools/watch.py --map maps/map_001.json    # Belirli harita
  python tools/watch.py --episodes 5               # 5 episode izle
  python tools/watch.py --delay 0.1                # Hız ayarı (saniye)
  python tools/watch.py --random                   # Eğitilmemiş (rastgele) ajan
"""
import os
import sys
import json
import time
import argparse
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from environment import GridEnvironment, ACTION_LABELS
from agent import DQLAgent

# Aksiyon → (satır_delta, sütun_delta) — grid_env.py ile aynı sıra: L=0 R=1 U=2 D=3
_STEP = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

# Q değerleri bu eşiğin altında yayılıyorsa (kararsız) → BFS devreye girer
_Q_CONFIDENCE = 10.0
# Bir hücre bu kadar kez ziyaret edilirse kalıcı BFS moduna geçilir (uzun döngü kırıcı)
# 2 = aynı hücreye 2. kez gelinince hemen BFS — optimal yolda hiçbir hücre iki kez ziyaret edilmez
_LOOP_THRESHOLD = 2
# Adım sayısı optimal BFS yolunun bu katını geçerse zorla BFS devreye girer
_STEP_BUDGET_RATIO = 3


def bfs_next_action(grid: np.ndarray, start: tuple, goal: tuple) -> int:
    """
    BFS ile start→goal en kısa yolu bul, ilk adımın aksiyonunu döndür.
    Engel yoksa garantili optimal hareket. Ulaşılamazsa -1.
    """
    if start == goal:
        return -1
    size = grid.shape[0]
    visited = {start}
    queue: deque = deque([(start, -1)])   # (pozisyon, ilk_aksiyon)

    while queue:
        pos, first_act = queue.popleft()
        for act, (dr, dc) in _STEP.items():
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            if grid[nr, nc] == 1:
                continue
            npos = (nr, nc)
            fa   = act if first_act == -1 else first_act
            if npos == goal:
                return fa
            if npos not in visited:
                visited.add(npos)
                queue.append((npos, fa))
    return -1  # ulaşılamaz


def smart_action(agent: DQLAgent, state: np.ndarray,
                 env: GridEnvironment, history: deque) -> int:
    """
    Hibrit aksiyon seçici:
    - Q değerleri yeterince ayrışıksa (spread ≥ _Q_CONFIDENCE):
        no-revisit Q-greedy kullan.
    - Q değerleri yakınsa (agent kararsız):
        BFS garantili yolu kullan → salınımı tamamen önler.
    """
    q_values = agent.get_q_values(state)
    q_spread  = float(q_values.max() - q_values.min())

    # ── Kararsız bölge: BFS devreye ──────────────────────────────────────────
    if q_spread < _Q_CONFIDENCE:
        bfs_act = bfs_next_action(env.grid, env.agent_pos, env.goal_pos)
        if bfs_act != -1:
            return bfs_act

    # ── Net Q sinyali: no-revisit Q-greedy ───────────────────────────────────
    sorted_actions = np.argsort(q_values)[::-1]
    best_valid = None
    for action in sorted_actions:
        dr, dc = _STEP[action]
        nr = env.agent_pos[0] + dr
        nc = env.agent_pos[1] + dc
        if not (0 <= nr < env.size and 0 <= nc < env.size):
            continue
        if env.grid[nr, nc] == 1:
            continue
        if best_valid is None:
            best_valid = action
        if (nr, nc) not in history:
            return action

    return best_valid if best_valid is not None else int(np.argmax(q_values))

# ─── Renkler (ANSI) ──────────────────────────────────────────────────────────
R  = "\033[0m"       # reset
BOLD = "\033[1m"
RED  = "\033[91m"
GRN  = "\033[92m"
YLW  = "\033[93m"
BLU  = "\033[94m"
MAG  = "\033[95m"
CYN  = "\033[96m"
GRY  = "\033[90m"
WHT  = "\033[97m"


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def render(env: GridEnvironment, action: int, q_values: np.ndarray,
           reward: float, step: int, episode: int,
           total_reward: float, epsilon: float,
           nav_mode: str = "Q", q_spread: float = 0.0) -> None:
    """Grid + istatistik panelini çiz."""

    size = env.size
    ar, ac = env.agent_pos
    gr, gc = env.goal_pos
    sr, sc = env.start_pos

    # ── Grid çiz ─────────────────────────────────────────────────────────────
    lines = []
    border = GRY + "+" + "──" * size + "─+" + R
    lines.append(border)

    for r in range(size):
        row = GRY + "│" + R
        for c in range(size):
            if (r, c) == (ar, ac):
                row += YLW + BOLD + " A" + R
            elif (r, c) == (gr, gc):
                row += GRN + BOLD + " G" + R
            elif (r, c) == (sr, sc):
                row += CYN + " S" + R
            elif env.grid[r, c] == 1:
                row += GRY + " █" + R
            else:
                row += " ."
        row += GRY + " │" + R
        lines.append(row)

    lines.append(border)

    # ── İstatistik paneli ────────────────────────────────────────────────────
    action_name = ACTION_LABELS.get(action, "?")
    action_arrows = {0: "←", 1: "→", 2: "↑", 3: "↓"}
    arrow = action_arrows.get(action, "?")

    q_str = "  ".join(
        f"{BLU if i == action else GRY}{ACTION_LABELS[i][0]}:{v:+.2f}{R}"
        for i, v in enumerate(q_values)
    )

    dist = abs(ar - gr) + abs(ac - gc)
    reward_color = GRN if reward > 0 else (RED if reward < -1 else GRY)
    mode_str = (f"{CYN}BFS{R}" if nav_mode == "BFS"
                else f"{MAG}Q{R}")

    panel = [
        f"",
        f"  {BOLD}Episode{R}  {WHT}{episode}{R}    {BOLD}Adım{R}  {WHT}{step}{R}    "
        f"{BOLD}ε{R}  {MAG}{epsilon:.3f}{R}    {BOLD}Mod{R}  {mode_str}  "
        f"{GRY}(spread:{q_spread:.1f}){R}",
        f"",
        f"  {BOLD}Aksiyon{R}   {YLW}{BOLD}{arrow} {action_name:<6}{R}",
        f"  {BOLD}Ödül{R}      {reward_color}{reward:+.1f}{R}",
        f"  {BOLD}ToplamÖdül{R} {WHT}{total_reward:+.1f}{R}",
        f"  {BOLD}Hedefe mesafe{R} {WHT}{dist}{R} adım",
        f"",
        f"  {BOLD}Q Değerleri{R}",
        f"  {q_str}",
        f"",
        f"  {GRY}S=Başlangıç  A=Ajan  G=Hedef  █=Engel{R}",
    ]

    clear()
    print("\n".join(lines))
    print("\n".join(panel))


def run_episode(env: GridEnvironment, agent: DQLAgent,
                episode: int, delay: float, training: bool,
                no_bfs: bool = False) -> dict:
    """Bir episode izle, sonucu döndür."""
    state = env.reset()
    total_reward = 0.0
    step = 0
    done = False

    # Döngü kırıcı geçmiş — saf RL'de daha uzun tutulur (no-revisit etkili olsun)
    history_len = 16 if no_bfs else 8
    history: deque = deque(maxlen=history_len)
    # Uzun döngü dedektörü: tüm episode boyunca ziyaret sayısı
    visit_counts: dict = {}
    bfs_forced = False

    # Adım bütçesi: BFS optimal yolunun _STEP_BUDGET_RATIO katını geçince zorla BFS
    _optimal_len = env._bfs_path_length(env.grid, env.agent_pos, env.goal_pos)
    _step_budget = max(_optimal_len * _STEP_BUDGET_RATIO, _optimal_len + 15)

    while not done:
        cur_pos = env.agent_pos
        visit_counts[cur_pos] = visit_counts.get(cur_pos, 0) + 1

        # Döngü tespiti — sadece BFS açıksa çalışır
        if not no_bfs and not bfs_forced:
            loop_detected   = visit_counts[cur_pos] >= _LOOP_THRESHOLD
            budget_exceeded = step >= _step_budget
            if loop_detected or budget_exceeded:
                bfs_forced = True

        history.append(cur_pos)

        if training and agent.epsilon > 0:
            action = agent.select_action(state, training=True)
            nav_mode = "ε-greedy"
        elif no_bfs:
            # Saf Q-network + ziyaret sayısı döngü kırıcı:
            # Geçerli aksiyonları Q sırasına göre sırala, hedef hücrenin
            # kaç kez ziyaret edildiğine bak. Hiç gidilmemiş varsa onu seç;
            # hepsi ziyaret edildiyse en az ziyaret edileni seç.
            q_vals_tmp = agent.get_q_values(state)
            sorted_acts = np.argsort(q_vals_tmp)[::-1]
            valid_actions = []  # (aksiyon, ziyaret_sayısı) — Q sırasına göre
            for a in sorted_acts:
                dr, dc = _STEP[a]
                nr, nc = cur_pos[0] + dr, cur_pos[1] + dc
                if not (0 <= nr < env.size and 0 <= nc < env.size):
                    continue
                if env.grid[nr, nc] == 1:
                    continue
                visits = visit_counts.get((nr, nc), 0)
                valid_actions.append((a, visits))

            if valid_actions:
                # Hiç gidilmemiş hücre varsa → en yüksek Q'lu olanı seç
                unvisited = [a for a, v in valid_actions if v == 0]
                if unvisited:
                    action = unvisited[0]
                else:
                    # Hepsi ziyaret edilmiş → en az ziyaret edilen, eşitlikte Q öncelikli
                    min_v = min(v for _, v in valid_actions)
                    action = next(a for a, v in valid_actions if v == min_v)
            else:
                action = int(np.argmax(q_vals_tmp))
            nav_mode = "Q-only"
        elif bfs_forced:
            bfs_act = bfs_next_action(env.grid, env.agent_pos, env.goal_pos)
            action = bfs_act if bfs_act != -1 else int(
                np.argmax(agent.get_q_values(state)))
            nav_mode = "BFS!"
        else:
            action = smart_action(agent, state, env, history)
            q_values_tmp = agent.get_q_values(state)
            nav_mode = ("BFS"
                        if float(q_values_tmp.max() - q_values_tmp.min()) < _Q_CONFIDENCE
                        else "Q")

        q_values = agent.get_q_values(state)
        q_spread = float(q_values.max() - q_values.min())

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        render(env, action, q_values, reward, step, episode,
               total_reward, agent.epsilon if training else 0.0,
               nav_mode=nav_mode, q_spread=q_spread)

        time.sleep(delay)
        state = next_state

    # Son kare — sonuç
    status = info.get("status", "?")
    status_color = GRN if info["reached_goal"] else RED
    print(f"\n  {status_color}{BOLD}{'✅ HEDEFE ULAŞTI!' if info['reached_goal'] else '❌ BAŞARISIZ — ' + status.upper()}{R}")
    print(f"  Toplam adım: {step}   Toplam ödül: {total_reward:+.1f}\n")
    time.sleep(1.2)

    return {
        "episode":       episode,
        "steps":         step,
        "total_reward":  round(total_reward, 2),
        "reached_goal":  info["reached_goal"],
        "status":        status,
    }


def print_summary(results: list) -> None:
    """Tüm episodeların özeti."""
    n           = len(results)
    successes   = sum(r["reached_goal"] for r in results)
    avg_reward  = sum(r["total_reward"] for r in results) / n
    avg_steps   = sum(r["steps"] for r in results) / n

    print(f"\n{'='*45}")
    print(f"  {BOLD}Özet — {n} Episode{R}")
    print(f"{'='*45}")
    print(f"  {'Ep':>4}  {'Adım':>5}  {'Ödül':>8}  {'Sonuç'}")
    print(f"  {'-'*38}")
    for r in results:
        icon = GRN + "✓" + R if r["reached_goal"] else RED + "✗" + R
        print(f"  {r['episode']:>4}  {r['steps']:>5}  {r['total_reward']:>+8.1f}  {icon}")
    print(f"  {'-'*38}")
    print(f"  {GRN}Başarı    : {successes}/{n} (%{successes/n*100:.0f}){R}")
    print(f"  Ort ödül  : {avg_reward:+.1f}")
    print(f"  Ort adım  : {avg_steps:.1f}")
    print(f"{'='*45}\n")


# ─── Zorluk seviyeleri ───────────────────────────────────────────────────────
# obstacle_ratio : haritadaki engel yoğunluğu
# min_path_length: BFS yolunun minimum adım sayısı (kısa/kolay haritalar elenir)
DIFFICULTY_PRESETS: dict = {
    "easy":    {"obstacle_ratio": 0.12, "min_path_length":  5},
    "medium":  {"obstacle_ratio": 0.20, "min_path_length": 12},
    "hard":    {"obstacle_ratio": 0.25, "min_path_length": 18},
    "extreme": {"obstacle_ratio": 0.30, "min_path_length": 22},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Ajanı canlı izle")
    parser.add_argument("--model",      default="models/best_model_final.pth",
                        help="Model dosyası (varsayılan: models/best_model_final.pth)")
    parser.add_argument("--map",        default=None,
                        help="Harita JSON dosyası (varsayılan: rastgele)")
    parser.add_argument("--episodes",   type=int, default=3,
                        help="İzlenecek episode sayısı (varsayılan: 3)")
    parser.add_argument("--delay",      type=float, default=0.18,
                        help="Adım arası bekleme süresi sn (varsayılan: 0.18)")
    parser.add_argument("--size",       type=int, default=15,
                        help="Grid boyutu (varsayılan: 15)")
    parser.add_argument("--random",     action="store_true",
                        help="Eğitilmemiş rastgele ajan kullan")
    parser.add_argument("--epsilon",    type=float, default=0.0,
                        help="İzleme sırasında epsilon (varsayılan: 0.0)")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "extreme"],
        default="easy",
        help=(
            "Harita zorluk seviyesi (varsayılan: easy)\n"
            "  easy    — engel %%12, min yol  5 adım\n"
            "  medium  — engel %%20, min yol 12 adım\n"
            "  hard    — engel %%25, min yol 18 adım\n"
            "  extreme — engel %%30, min yol 22 adım"
        ),
    )
    parser.add_argument(
        "--bfs",
        action="store_true",
        dest="use_bfs",
        help="BFS yardımını ve döngü kırıcıyı aç (varsayılan: kapalı — saf RL politikası)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # ── Ortamı hazırla ───────────────────────────────────────────────────────
    diff_cfg = DIFFICULTY_PRESETS[args.difficulty]

    if args.map:
        env = GridEnvironment(size=args.size, random_maps=False)
        with open(args.map, encoding="utf-8") as f:
            payload = json.load(f)
        env.load_from_api_payload(payload)
        print(f"  Harita yüklendi: {args.map}")
    else:
        env = GridEnvironment(
            size=args.size,
            random_maps=True,
            obstacle_ratio=diff_cfg["obstacle_ratio"],
            min_path_length=diff_cfg["min_path_length"],
        )

    # ── Ajanı yükle ──────────────────────────────────────────────────────────
    # Checkpoint'ten hidden_size oku — ağ mimarisi uyumsuzluğunu önler
    hidden_size = 64
    if not args.random and os.path.exists(args.model):
        import torch as _torch
        _ckpt = _torch.load(args.model, map_location="cpu", weights_only=False)
        if "config" in _ckpt and "hidden_size" in _ckpt["config"]:
            hidden_size = _ckpt["config"]["hidden_size"]
        else:
            # Eski checkpoint: ağırlık şeklinden çıkar
            w = _ckpt["q_network_state"].get("feature.0.weight")
            if w is not None:
                hidden_size = w.shape[0]

    agent = DQLAgent(state_size=env.state_size, action_size=env.action_size,
                     hidden_size=hidden_size)

    if not args.random:
        loaded = agent.load(args.model)
        if not loaded:
            print(f"  [!] Model bulunamadı: {args.model}")
            print(f"  Rastgele ajan kullanılıyor...")
        agent.epsilon = args.epsilon  # Küçük epsilon döngü kırar
    else:
        agent.epsilon = 1.0           # Saf keşif — rastgele hareketler
        print("  [!] Rastgele ajan modu")

    diff_label = args.difficulty.upper()
    diff_info  = (f"engel %{int(diff_cfg['obstacle_ratio']*100)}  "
                  f"min_yol {diff_cfg['min_path_length']} adım")
    mode = "🎲 Rastgele Ajan" if args.random else f"🧠 Eğitilmiş Model ({args.model})"
    nav_label  = (f"{CYN}Hibrit Q+BFS{R}" if args.use_bfs
                  else f"{GRN}Saf RL (Q-network){R}")
    print(f"\n  {BOLD}Ajan İzleme{R} — {mode}")
    print(f"  Grid: {args.size}×{args.size}   Episodes: {args.episodes}   "
          f"Hız: {args.delay}s/adım   ε: {agent.epsilon:.3f}")
    print(f"  Zorluk: {BOLD}{diff_label}{R}  ({diff_info})")
    print(f"  Navigasyon: {nav_label}")
    print(f"\n  {GRY}Başlamak için Enter'a bas...{R}")
    input()

    # ── Episodeları çalıştır ─────────────────────────────────────────────────
    results = []
    for ep in range(1, args.episodes + 1):
        if args.map:
            env.load_from_api_payload(payload)   # aynı haritayı yenile
        result = run_episode(env, agent, ep, args.delay,
                             training=(agent.epsilon > 0),
                             no_bfs=not args.use_bfs)
        results.append(result)

    print_summary(results)
