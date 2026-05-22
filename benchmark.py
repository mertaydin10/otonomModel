import os
import sys
import math
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ==========================================
# 1. ORTAM (ENVIRONMENT) VE DESTEK SINIFLARI
# ==========================================
class DynamicObstacle:
    def __init__(self, x, y, vx, vy, grid_size):
        self.x, self.y = float(x), float(y)
        self.vx, self.vy = float(vx), float(vy)
        self.grid_size = grid_size
    def step(self):
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x >= self.grid_size - 1: self.vx *= -1
        if self.y < 0 or self.y >= self.grid_size - 1: self.vy *= -1
    @property
    def grid_pos(self): return (int(round(self.x)), int(round(self.y)))
    def normalized_velocity(self, max_speed): return (self.vx / max_speed, self.vy / max_speed)

class MapGenerator:
    def __init__(self, grid_size, n_static):
        self.grid_size, self.n_static = grid_size, n_static

    def has_path(self, start, goal, static_obstacles):
        """BFS ile başlangıç ve hedef arasında en az bir yol olduğunu garanti eder."""
        queue = deque([start])
        visited = {start}
        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = curr[0] + dx, curr[1] + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) not in static_obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False

    def generate(self, map_type=None):
        static_obs = set()
        while len(static_obs) < self.n_static:
            static_obs.add((random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)))
        return static_obs, "procedural"

    def get_start_and_goal(self, static_obstacles):
        """%100 çözülebilir bir başlangıç ve hedef ikilisi bulana kadar dener."""
        attempts = 0
        while True:
            attempts += 1
            s = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            g = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if s not in static_obstacles and g not in static_obstacles and s != g:
                if math.hypot(s[0]-g[0], s[1]-g[1]) > self.grid_size * 0.4:
                    if self.has_path(s, g, static_obstacles):
                        return s, g
            if attempts > 50:
                static_obstacles.clear()
                while len(static_obstacles) < self.n_static:
                    static_obstacles.add((random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)))
                attempts = 0

class AutonomousDriverEnv:
    def __init__(self, grid_size=12, n_static=4, n_dynamic=0, max_steps=150, view_radius=7):
        self.grid_size = grid_size
        self.n_static = n_static
        self.n_dynamic = n_dynamic
        self.max_steps = max_steps
        self.view_radius = view_radius
        self.map_gen = MapGenerator(grid_size, n_static)

        self.agent_pos = None
        self.goal_pos = None
        self.static_obs = set()
        self.dynamic_obs = []
        self.visit_map = None
        self.action_history = None
        self.steps = 0
        self.prev_dist = 0
        self.map_type_current = ""

    def reset(self):
        self.static_obs, self.map_type_current = self.map_gen.generate()
        self.agent_pos, self.goal_pos = self.map_gen.get_start_and_goal(self.static_obs)

        self.dynamic_obs = []
        for _ in range(self.n_dynamic):
            while True:
                dx = random.randint(0, self.grid_size-1)
                dy = random.randint(0, self.grid_size-1)
                if (dx, dy) not in self.static_obs and (dx, dy) != self.agent_pos and (dx, dy) != self.goal_pos:
                    speed = 0.5
                    if random.choice([True, False]):
                        vx = random.choice([-speed, speed])
                        vy = 0.0
                    else:
                        vx = 0.0
                        vy = random.choice([-speed, speed])
                    self.dynamic_obs.append(DynamicObstacle(dx, dy, vx, vy, self.grid_size))
                    break

        self.visit_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.visit_map[self.agent_pos[0], self.agent_pos[1]] += 1

        self.action_history = deque([[0.0]*5 for _ in range(8)], maxlen=8)
        self.steps = 0
        self.prev_dist = math.hypot(self.agent_pos[0]-self.goal_pos[0], self.agent_pos[1]-self.goal_pos[1])

        info = {"map_type": self.map_type_current}
        return self._get_obs(), info

    def step(self, action):
        self.steps += 1
        reward = -0.05
        terminated = False
        truncated = False
        is_success = False

        action_oh = [0.0]*5
        action_oh[action] = 1.0
        self.action_history.append(action_oh)

        for dyn in self.dynamic_obs:
            dyn.step()

        dx, dy = 0, 0
        if action == 0: dy = -1
        elif action == 1: dy = 1
        elif action == 2: dx = -1
        elif action == 3: dx = 1

        move_attempted = (dx != 0 or dy != 0)
        is_waiting = (action == 4)

        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        hit_static = False
        if move_attempted:
            if not (0 <= nx < self.grid_size and 0 <= ny < self.grid_size):
                hit_static = True
            elif (nx, ny) in self.static_obs:
                hit_static = True

        if hit_static:
            reward -= 50.0
            terminated = True
            self.visit_map[self.agent_pos[0], self.agent_pos[1]] += 1
        elif move_attempted:
            self.agent_pos = (nx, ny)
            visits = self.visit_map[nx, ny]
            if visits == 0:
                reward += 0.3
            else:
                reward -= 0.05 * visits
            self.visit_map[nx, ny] += 1
        elif is_waiting:
            pass

        if not terminated:
            min_dyn_dist = float('inf')
            for dyn in self.dynamic_obs:
                dist = math.hypot(self.agent_pos[0] - dyn.x, self.agent_pos[1] - dyn.y)
                if dist < min_dyn_dist:
                    min_dyn_dist = dist

            if min_dyn_dist <= 0.8:
                reward -= 50.0
                terminated = True
            elif min_dyn_dist <= 1.5:
                reward -= 0.2

        curr_dist = math.hypot(self.agent_pos[0]-self.goal_pos[0], self.agent_pos[1]-self.goal_pos[1])
        reward += (self.prev_dist - curr_dist) * 1.5
        self.prev_dist = curr_dist

        if not terminated and self.agent_pos == self.goal_pos:
            reward += 100.0
            terminated = True
            is_success = True

        if not terminated and self.steps >= self.max_steps:
            truncated = True

        info = {
            "is_success": is_success,
            "map_type": self.map_type_current
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _get_obs(self):
        obs = []
        dirs = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]
        dyn_positions_dict = {dyn.grid_pos: dyn for dyn in self.dynamic_obs}

        for dx, dy in dirs:
            hit_data = [1.0, 0.0, 0.0, 0.0]
            for step in range(1, self.view_radius + 1):
                rx, ry = self.agent_pos[0] + dx*step, self.agent_pos[1] + dy*step
                if rx < 0 or rx >= self.grid_size or ry < 0 or ry >= self.grid_size or (rx, ry) in self.static_obs:
                    hit_data = [step/self.view_radius, 1.0, 0.0, 0.0]
                    break
                if (rx, ry) in dyn_positions_dict:
                    dyn = dyn_positions_dict[(rx, ry)]
                    nvx, nvy = dyn.normalized_velocity(1.0)
                    hit_data = [step/self.view_radius, 0.5, nvx, nvy]
                    break
            obs.extend(hit_data)

        delta_x = (self.goal_pos[0] - self.agent_pos[0]) / self.grid_size
        delta_y = (self.goal_pos[1] - self.agent_pos[1]) / self.grid_size
        dist_norm = math.hypot(delta_x, delta_y) / math.sqrt(2)
        angle = math.atan2(delta_y, delta_x)
        obs.extend([delta_x, delta_y, dist_norm, math.sin(angle), math.cos(angle)])

        max_vis = max(1.0, np.max(self.visit_map))
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = self.agent_pos[0]+dx, self.agent_pos[1]+dy
                if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                    obs.append(self.visit_map[px, py] / max_vis)
                else:
                    obs.append(1.0)

        for act_arr in self.action_history:
            obs.extend(act_arr)

        return np.array(obs, dtype=np.float32)

# ==========================================
# 2. AGENT DEFINITIONS
# ==========================================
from agent.ppo_agent import PPOAgent
from agent.dql_agent import DQLAgent

def get_dqn_state(env, state_size):
    s = max(env.grid_size - 1, 1)
    col, row = env.agent_pos[0], env.agent_pos[1]
    gcol, grow = env.goal_pos[0], env.goal_pos[1]
    
    imm = np.zeros(4, dtype=np.float32)
    dist = np.zeros(4, dtype=np.float32)
    
    # DQN direction layout: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN (dc, dr)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def is_blocked(cx, cy):
        if not (0 <= cx < env.grid_size and 0 <= cy < env.grid_size):
            return True
        if (cx, cy) in env.static_obs:
            return True
        for dyn in env.dynamic_obs:
            if int(round(dyn.x)) == cx and int(round(dyn.y)) == cy:
                return True
        return False
        
    for act in range(4):
        dc, dr = dirs[act]
        nx, ny = col + dc, row + dr
        if is_blocked(nx, ny):
            imm[act] = 1.0
            
        cx, cy = col, row
        d = 0
        while True:
            cx, cy = cx + dc, cy + dr
            if not (0 <= cx < env.grid_size and 0 <= cy < env.grid_size):
                break
            d += 1
            if is_blocked(cx, cy):
                break
        dist[act] = d / s
        
    if state_size == 12:
        return np.array([
            col / s, row / s,
            gcol / s, grow / s,
            imm[0], imm[1], imm[2], imm[3],
            dist[0], dist[1], dist[2], dist[3]
        ], dtype=np.float32)
    elif state_size == 16:
        nearest_dx = 0.0
        nearest_dy = 0.0
        nearest_move_dx = 0.0
        nearest_move_dy = 0.0
        
        if env.dynamic_obs:
            min_dist = float('inf')
            nearest_obs = None
            for dyn in env.dynamic_obs:
                d = abs(int(round(dyn.x)) - col) + abs(int(round(dyn.y)) - row)
                if d < min_dist:
                    min_dist = d
                    nearest_obs = dyn
            if nearest_obs is not None:
                nearest_dx = (int(round(nearest_obs.x)) - col) / s
                nearest_dy = (int(round(nearest_obs.y)) - row) / s
                nearest_move_dx = float(np.sign(nearest_obs.vx))
                nearest_move_dy = float(np.sign(nearest_obs.vy))
                
        return np.array([
            col / s, row / s,
            gcol / s, grow / s,
            imm[0], imm[1], imm[2], imm[3],
            dist[0], dist[1], dist[2], dist[3],
            nearest_dx, nearest_dy,
            nearest_move_dx, nearest_move_dy
        ], dtype=np.float32)
        
    return np.zeros(state_size, dtype=np.float32)

def run_benchmark(model_key, env, num_episodes=300):
    model_path = os.path.join(project_root, "models", model_key)
    
    is_dql = model_key.endswith(".pth")
    
    # Dynamically scale view_radius based on evaluated model for tomorrow's experiments!
    lower_key = model_key.lower()
    if any(x in lower_key for x in ["radius3", "radius_3", "_r3", "sweetspot3"]):
        env.view_radius = 3
    elif any(x in lower_key for x in ["radius4", "radius_4", "_r4", "sweetspot"]):
        env.view_radius = 4
    elif any(x in lower_key for x in ["radius5", "radius_5", "_r5", "sweetspot5"]):
        env.view_radius = 5
    elif any(x in lower_key for x in ["radius6", "radius_6", "_r6", "sweetspot6"]):
        env.view_radius = 6
    else:
        env.view_radius = 7
        
    if is_dql:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        cfg = checkpoint["config"]
        state_size = cfg.get("state_size", 16)
        action_size = cfg.get("action_size", 4)
        hidden_size = cfg.get("hidden_size", 256)
        
        agent = DQLAgent(
            state_size=state_size,
            action_size=action_size,
            hidden_size=hidden_size
        )
        agent.load(model_path)
    else:
        agent = PPOAgent(state_size=102, action_size=5)
        agent.load(model_path)

    success_count = 0
    total_steps = 0
    total_reward = 0
    collisions = 0
    timeouts = 0

    print(f"\n🧪 Testing model: {model_key} over {num_episodes} identical episodes... (View Radius: {env.view_radius})")

    for ep in range(num_episodes):
        # Guarantee 100% identical seeds per episode index (new map pool seed: 8888)
        random.seed(8888 + ep)
        np.random.seed(8888 + ep)

        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0

        while not done:
            if is_dql:
                # 1. Correctly generate state vector using the expected DQN layout
                sliced_obs = get_dqn_state(env, agent.state_size)
                dql_action = agent.select_action(sliced_obs, training=False)
                # 2. Correctly map action IDs from DQN structure (LEFT/RIGHT/UP/DOWN) to PPO
                if dql_action == 0: action = 2      # LEFT
                elif dql_action == 1: action = 3    # RIGHT
                elif dql_action == 2: action = 0    # UP
                elif dql_action == 3: action = 1    # DOWN
                else: action = 4
            else:
                action = agent.act(obs)
                
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated

        if info.get("is_success", False):
            success_count += 1
        elif ep_steps >= env.max_steps:
            timeouts += 1
        else:
            collisions += 1

        total_steps += ep_steps
        total_reward += ep_reward

        if (ep + 1) % 50 == 0:
            print(f"  [Progress] {ep + 1}/{num_episodes} runs complete... Current Success Rate: %{(success_count / (ep+1)) * 100:.1f}")

    success_rate = (success_count / num_episodes) * 100
    avg_steps = total_steps / num_episodes
    avg_reward = total_reward / num_episodes
    collision_rate = (collisions / num_episodes) * 100
    timeout_rate = (timeouts / num_episodes) * 100

    return {
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "success_count": success_count,
        "collisions": collisions,
        "timeouts": timeouts
    }

if __name__ == "__main__":
    MODELS = [
        ("ppo_v3",              "PPO v3"),
        ("ppo_stage_4_hardcore","PPO v4 Hardcore"),
        ("ppo_sweetspot",       "Sweet Spot 1 (Stg3)"),
        ("ppo_sweetspot_2",     "Sweet Spot 2 (Final)"),
        ("ppo_sweetspot_3",     "Sweet Spot 3 (NEW)"),
    ]

    GRIDS = [
        {"grid_size": 15, "n_static": 12, "n_dynamic": 4,  "max_steps": 250, "label": "15x15 (Standart)"},
        {"grid_size": 21, "n_static": 25, "n_dynamic": 8,  "max_steps": 350, "label": "21x21 (Orta)"},
        {"grid_size": 31, "n_static": 50, "n_dynamic": 17, "max_steps": 500, "label": "31x31 (Ekstrem)"},
    ]

    all_results = {}  # {grid_label: {model_label: results}}

    for g in GRIDS:
        env = AutonomousDriverEnv(
            grid_size=g["grid_size"], n_static=g["n_static"],
            n_dynamic=g["n_dynamic"], max_steps=g["max_steps"], view_radius=7
        )
        print(f"\n{'🔥'*20}")
        print(f"  GRID: {g['label']}  |  Statik: {g['n_static']}  |  Dinamik: {g['n_dynamic']}")
        print(f"{'🔥'*20}")

        grid_results = {}
        for model_key, model_label in MODELS:
            r = run_benchmark(model_key, env, num_episodes=300)
            grid_results[model_label] = r
        all_results[g["label"]] = grid_results

    # ─── Sonuç Tabloları ─────────────────────────────────────────────
    col_w = 18
    header_labels = [ml for _, ml in MODELS]
    table_w = 25 + 3 + (col_w + 3) * len(MODELS)

    for grid_label, grid_results in all_results.items():
        print(f"\n{'='*table_w}")
        print(f"📊  {grid_label}  —  300 Identical Maps")
        print(f"{'='*table_w}")
        hdr = f"{'Metric':<25} | " + " | ".join(f"{h:<{col_w}}" for h in header_labels)
        print(hdr)
        print("-" * table_w)

        rows = [
            ("Success Rate %",  "success_rate"),
            ("Collision Rate %","collision_rate"),
            ("Timeout Rate %",  "timeout_rate"),
            ("Avg Steps",       "avg_steps"),
            ("Avg Reward",      "avg_reward"),
        ]
        for label, key in rows:
            vals = []
            for _, ml in MODELS:
                v = grid_results[ml][key]
                if "rate" in key.lower():
                    vals.append(f"%{v:<{col_w-1}.1f}")
                else:
                    vals.append(f"{v:<{col_w}.1f}")
            print(f"{label:<25} | " + " | ".join(vals))
        print("=" * table_w)

        winner = max(grid_results.keys(), key=lambda k: grid_results[k]["success_rate"])
        print(f"🏆 {grid_label} Şampiyonu: {winner} — %{grid_results[winner]['success_rate']:.1f}")

    # ─── Genel Özet ───────────────────────────────────────────────────
    print(f"\n\n{'🏁'*30}")
    print("📊  GENEL ÖZET  —  TÜM GRİD BOYUTLARINDA BAŞARI ORANLARI")
    print(f"{'🏁'*30}")
    summary_hdr = f"{'Model':<25} | " + " | ".join(f"{g['label']:<{col_w}}" for g in GRIDS)
    print(summary_hdr)
    print("-" * (25 + 3 + (col_w + 3) * len(GRIDS)))
    for _, ml in MODELS:
        vals = []
        for g in GRIDS:
            sr = all_results[g["label"]][ml]["success_rate"]
            vals.append(f"%{sr:<{col_w-1}.1f}")
        print(f"{ml:<25} | " + " | ".join(vals))
    print("=" * (25 + 3 + (col_w + 3) * len(GRIDS)))

    # Genel şampiyon (tüm gridlerdeki toplam başarı ortalaması)
    avg_scores = {}
    for _, ml in MODELS:
        total = sum(all_results[g["label"]][ml]["success_rate"] for g in GRIDS)
        avg_scores[ml] = total / len(GRIDS)
    overall_winner = max(avg_scores, key=avg_scores.get)
    print(f"\n🏆🏆🏆 GENEL ŞAMPİYON: {overall_winner} — Ortalama %{avg_scores[overall_winner]:.1f} başarı oranı!")
    print("=" * (25 + 3 + (col_w + 3) * len(GRIDS)))

