"""
api/main.py — FastAPI WebSocket AI Servisi

Endpointler:
  GET  /health           → Servis sağlık kontrolü
  POST /train            → Arka planda eğitim başlat
  GET  /train/status     → Eğitim durumunu sorgula
  POST /train/stop       → Eğitimi durdur
  POST /infer            → Tek adım inference
  GET  /model/stats      → Model istatistikleri
  WS   /ws/simulate      → Gerçek zamanlı simülasyon (Spring Boot uyumlu)
  POST /maps/load        → GameMapDTO yükle ve ortamı sıfırla

Simülasyon WebSocket protokolü:
  İstemci → Sunucu (AgentTickDTO formatı):
  {
    "map_name":  "harita1",
    "agent_pos": {"x": -5, "y": 0},
    "goal_pos":  {"x":  5, "y": 0},
    "grid":      [[0,1,...], ...],   // opsiyonel
    "state":     [0.1, 0.2, ...]    // 12 elemanlı, opsiyonel
  }

  Sunucu → İstemci (SimulationResponseDTO formatı):
  {
    "action":       2,
    "action_label": "UP",
    "q_values":     [1.2, 0.3, -0.5, 0.8],
    "reward":       1.0,
    "done":         false,
    "reached_goal": false,
    "epsilon":      0.15,
    "episode":      420,
    "agent_pos":    {"x": -5, "y": 1},
    "steps":        12
  }
"""
import os
import sys
import json
import threading
import numpy as np
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import GridEnvironment, ACTION_LABELS
from agent import DQLAgent
from training.train import DEFAULT_CONFIG

# ─── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Otonom Sürüş AI Servisi",
    description="DQL tabanlı otonom sürüş ajanı — FastAPI + PyTorch",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Durum ─────────────────────────────────────────────────────────────

import torch
from agent.ppo_agent import PPOAgent
from collections import deque

# ─── Model Yükleme ve Konfigürasyon ───
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "best_model_v4.pth"
)
DEFAULT_SIZE = 15  # best_model_v4 genelde 15x15 için eğitildi

# PPO mu yoksa DQN mu otomatik olarak tespit et ve yükle
IS_PPO = os.path.exists(os.path.join(MODEL_PATH, "policy.pth")) or "ppo" in MODEL_PATH.lower()

if IS_PPO:
    print(f"[INIT] PPO Hardcore Model tespit edildi! Model: {MODEL_PATH}")
    env = GridEnvironment(size=DEFAULT_SIZE, random_maps=True, state_size=102)
    agent = PPOAgent(state_size=102, action_size=5)
    try:
        agent.load(MODEL_PATH)
    except Exception as _load_err:
        print(f"[WARN] PPO model yüklenemedi: {_load_err}")
else:
    print(f"[INIT] DQN Model yükleniyor: {MODEL_PATH}")
    # Checkpoint'i yükleyip içindeki gizli katman ve durum vektörü boyutlarını dinamik oku
    state_size = 16  # Varsayılan
    hidden_size = 256  # Varsayılan
    action_size = 4  # Varsayılan
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if "config" in checkpoint:
            state_size = checkpoint["config"].get("state_size", state_size)
            hidden_size = checkpoint["config"].get("hidden_size", hidden_size)
            action_size = checkpoint["config"].get("action_size", action_size)
            print(f"[INIT] Checkpoint konfigürasyonu okundu: state_size={state_size}, hidden_size={hidden_size}, action_size={action_size}")
    except Exception as _read_err:
        print(f"[WARN] Checkpoint okunamadı, varsayılan boyutlar kullanılacak: {_read_err}")
        
    env = GridEnvironment(size=DEFAULT_SIZE, random_maps=True, state_size=state_size)
    agent = DQLAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size
    )
    try:
        agent.load(MODEL_PATH)
    except Exception as _load_err:
        print(f"[WARN] DQN model yüklenemedi, yeni ağırlıklarla başlanıyor: {_load_err}")

training_state: Dict[str, Any] = {
    "running": False,
    "episode": 0,
    "max_ep": 0,
    "last_reward": None,
    "success_rate": 0.0,
    "epsilon": 1.0,
    "error": None,
}


# ─── Pydantic Modeller ────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    episodes: int = 2000
    grid_size: int = 15
    obstacle_ratio: float = 0.15
    learning_rate: float = 0.001
    gamma: float = 0.95
    random_maps: bool = True


class InferRequest(BaseModel):
    state: list[float]


class SelectModelRequest(BaseModel):
    model_key: str


class DynamicObstacleDTO(BaseModel):
    """Frontend'den gelen dinamik engel — payload format"""
    id: int
    pos: Dict[str, int]              # {x, y}
    velocity: Dict[str, int]         # {vx, vy}
    range: Optional[int] = None
    type: str = "linear-h"           # "linear-h" | "linear-v" | "random"


class ObstaclesDTO(BaseModel):
    """Statik + dinamik engeller"""
    static: list[dict] = []
    dynamic: list[DynamicObstacleDTO] = []


class MapPayload(BaseModel):
    map_name: str
    grid_size: Dict[str, int]
    start_pos: Dict[str, int]
    target_pos: Dict[str, int]
    obstacles: ObstaclesDTO


# ─── Endpointler ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
        "device": str(agent.device),
        "episode": int(agent.episode_count),
        "state_size": int(env.state_size),
    }


@app.get("/models")
async def get_models():
    """Mevcut tüm otonom sürüş modellerini ve aktif olanı listele."""
    # Models klasöründeki tüm pth dosyalarını listele
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    
    available_models = [
        {"key": "ppo_stage_4_hardcore", "name": "PPO (ppo_stage_4_hardcore)", "type": "PPO"}
    ]
    
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".pth"):
                key = f.replace(".pth", "")
                name = f"DQN ({key})"
                available_models.append({"key": key, "name": name, "type": "DQN"})
                
    # Aktif model ismini bul
    active_key = os.path.basename(MODEL_PATH).replace(".pth", "")
    if "ppo_stage_4" in MODEL_PATH:
        active_key = "ppo_stage_4_hardcore"
        
    return {
        "models": available_models,
        "active_model": active_key
    }


@app.post("/model/select")
async def select_model(req: SelectModelRequest):
    """Canlı olarak otonom sürüş modelini değiştir."""
    global MODEL_PATH, IS_PPO, env, agent
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if req.model_key == "ppo_stage_4_hardcore":
            new_path = os.path.join(project_root, "models", "ppo_stage_4_hardcore")
        else:
            new_path = os.path.join(project_root, "models", f"{req.model_key}.pth")
            
        if not os.path.exists(new_path):
            raise HTTPException(status_code=404, detail=f"Model dosyası bulunamadı: {new_path}")
            
        MODEL_PATH = new_path
        IS_PPO = os.path.exists(os.path.join(MODEL_PATH, "policy.pth")) or "ppo" in MODEL_PATH.lower()
        
        if IS_PPO:
            print(f"[DYNAMIC CHANGE] PPO modeline geçiliyor: {MODEL_PATH}")
            env = GridEnvironment(size=DEFAULT_SIZE, random_maps=True, state_size=102)
            agent = PPOAgent(state_size=102, action_size=5)
            agent.load(MODEL_PATH)
        else:
            print(f"[DYNAMIC CHANGE] DQN modeline geçiliyor: {MODEL_PATH}")
            state_size = 16
            hidden_size = 256
            action_size = 4
            
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
            if "config" in checkpoint:
                state_size = checkpoint["config"].get("state_size", state_size)
                hidden_size = checkpoint["config"].get("hidden_size", hidden_size)
                action_size = checkpoint["config"].get("action_size", action_size)
                
            env = GridEnvironment(size=DEFAULT_SIZE, random_maps=True, state_size=state_size)
            agent = DQLAgent(
                state_size=state_size,
                action_size=action_size,
                hidden_size=hidden_size
            )
            agent.load(MODEL_PATH)
            
        print(f"[DYNAMIC CHANGE] Model değişimi başarılı! Aktif model: {req.model_key}")
        return {
            "status": "success",
            "active_model": req.model_key,
            "type": "PPO" if IS_PPO else "DQN",
            "state_size": env.state_size
        }
    except Exception as err:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model yüklenemedi: {str(err)}")


@app.post("/maps/load")
async def load_map(payload: MapPayload):
    """Frontend'den gelen harita + dinamik engelleri yükle."""
    global env
    try:
        print(f"[LOAD] Gelen payload: map_name={payload.map_name}, grid_size={payload.grid_size}, obstacles_count={len(payload.obstacles.static) if payload.obstacles else 0}")
        # Use model_dump() to ensure dict format (avoids Pydantic object issues)
        payload_dict = payload.model_dump()

        # Fresh GridEnvironment instance
        size = payload_dict["grid_size"]["x"]
        env = GridEnvironment(size=size, random_maps=False, state_size=agent.state_size)

        # Harita yükle — inline implementation
        half = size // 2

        def cart_to_idx(x, y):
            return (half - y, x + half)

        # Static obstacles
        grid = np.zeros((size, size), dtype=np.int8)
        for obs in payload_dict["obstacles"]["static"]:
            r, c = cart_to_idx(obs["x"], obs["y"])
            if 0 <= r < size and 0 <= c < size:
                grid[r, c] = 1

        # Set grid and positions
        sp = payload_dict["start_pos"]
        tp = payload_dict["target_pos"]
        start = cart_to_idx(sp["x"], sp["y"])
        goal = cart_to_idx(tp["x"], tp["y"])

        env._generate_from_data(grid, start, goal)

        # Debug: Harita görüntüsü
        print(f"[MAP] Harita yüklendi: {size}x{size}")
        print(f"[MAP] Baslangi (grid): {start} -> Cartesian: ({start[1]-half}, {half-start[0]})")
        print(f"[MAP] Hedef (grid): {goal} -> Cartesian: ({goal[1]-half}, {half-goal[0]})")
        print(f"[MAP] Grid (1=engel, 0=bos):")
        for r in range(size):
            row_str = "".join("#" if grid[r, c] == 1 else "." for c in range(size))
            marker = ""
            if tuple([r, start[1]]) == tuple(start): marker += " START"
            if tuple([r, goal[1]]) == tuple(goal): marker += " GOAL"
            print(f"  {row_str}{marker}")

        # Dinamik engelleri yükle
        if payload_dict["obstacles"] and payload_dict["obstacles"]["dynamic"]:
            try:
                # Ensure all dynamic obstacles are pure dicts, not Pydantic objects
                dynamic_list = payload_dict["obstacles"]["dynamic"]
                dynamic_dicts = []
                for obs in dynamic_list:
                    if isinstance(obs, dict):
                        dynamic_dicts.append(obs)
                    elif hasattr(obs, 'model_dump'):
                        dynamic_dicts.append(obs.model_dump())
                    elif hasattr(obs, '__dict__'):
                        dynamic_dicts.append(obs.__dict__)
                env.load_dynamic_obstacles_from_api(dynamic_dicts)
            except Exception as dyn_err:
                # Fallback: dynamic obstacles optional (log but continue)
                print(f"[WARN] Dinamik engel yüklenemedi: {dyn_err}")
                import traceback
                traceback.print_exc()
                pass

        # Episode'i başlat
        env.reset()

        return {
            "status": "ok",
            "map_name": payload.map_name,
            "grid_size": payload.grid_size,
            "start": {"row": int(env.start_pos[0]), "col": int(env.start_pos[1])},
            "goal": {"row": int(env.goal_pos[0]), "col": int(env.goal_pos[1])},
            "dynamic_obstacles": len(env.dynamic_obstacles),
        }
    except Exception as exc:
        print(f"[ERROR] /maps/load exception: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/infer")
async def infer(req: InferRequest):
    """Tek adım inference — durum vektörü → aksiyon + Q değerleri."""
    if len(req.state) != env.state_size:
        raise HTTPException(
            status_code=422,
            detail=f"Durum vektörü boyutu {env.state_size} olmalı, {len(req.state)} geldi.",
        )
    state = np.array(req.state, dtype=np.float32)
    action = agent.select_action(state, training=False)
    q_values = agent.get_q_values(state)
    return {
        "action": action,
        "action_label": ACTION_LABELS[action],
        "q_values": q_values.tolist(),
    }


@app.get("/model/stats")
async def model_stats():
    return agent.get_stats()


# ─── Eğitim ───────────────────────────────────────────────────────────────────

def _run_training(config: dict) -> None:
    global training_state, env, agent

    training_state.update(
        running=True, error=None, episode=0, max_ep=config["max_episodes"]
    )

    try:
        from collections import deque

        env = GridEnvironment(
            size=config["grid_size"],
            obstacle_ratio=config["obstacle_ratio"],
            max_steps=config["max_steps"],
            random_maps=config["random_maps"],
        )
        agent = DQLAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            hidden_size=config.get("hidden_size", 128),
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            epsilon_start=config["epsilon_start"],
            epsilon_min=config["epsilon_min"],
            epsilon_decay=config["epsilon_decay"],
            batch_size=config["batch_size"],
            buffer_capacity=config["buffer_capacity"],
            target_update=config["target_update"],
        )

        window = deque(maxlen=100)
        success_win = deque(maxlen=100)
        best_avg = float("-inf")

        for ep in range(1, config["max_episodes"] + 1):
            if not training_state["running"]:
                break

            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                if agent.total_steps % 4 == 0:
                    agent.train_step()
                state = next_state
                total_reward += reward

            agent.decay_epsilon()
            if ep % config["target_update"] == 0:
                agent.update_target_network()

            window.append(total_reward)
            success_win.append(int(info["reached_goal"]))
            avg = float(np.mean(window))
            if avg > best_avg and len(window) == 100:
                best_avg = avg
                agent.save(config["model_path"])

            training_state.update(
                episode=ep,
                last_reward=round(total_reward, 2),
                success_rate=round(float(np.mean(success_win)), 3),
                epsilon=round(agent.epsilon, 4),
            )

    except Exception as exc:
        training_state["error"] = str(exc)
    finally:
        training_state["running"] = False


@app.post("/train")
async def start_training(req: TrainRequest, _bg: BackgroundTasks):
    if training_state["running"]:
        raise HTTPException(status_code=409, detail="Eğitim zaten çalışıyor.")

    config = DEFAULT_CONFIG.copy()
    config.update(
        max_episodes=req.episodes,
        grid_size=req.grid_size,
        obstacle_ratio=req.obstacle_ratio,
        learning_rate=req.learning_rate,
        gamma=req.gamma,
        random_maps=req.random_maps,
        model_path=MODEL_PATH,
    )
    threading.Thread(target=_run_training, args=(config,), daemon=True).start()
    return {"status": "started", "config": config}


@app.get("/train/status")
async def training_status():
    return {
        "running": training_state["running"],
        "episode": training_state["episode"],
        "max_episodes": training_state["max_ep"],
        "last_reward": training_state["last_reward"],
        "success_rate": training_state["success_rate"],
        "epsilon": training_state["epsilon"],
        "error": training_state["error"],
        "progress": round(
            training_state["episode"] / max(training_state["max_ep"], 1), 3
        ),
    }


@app.post("/train/stop")
async def stop_training():
    training_state["running"] = False
    return {"status": "stopped", "episode": training_state["episode"]}



# ─── WebSocket Simülasyon ─────────────────────────────────────────────────────

import math

def get_ppo_observation(env, view_radius=7):
    # PPO visit_map ve action_history'yi GridEnvironment nesnesine bağla
    if not hasattr(env, "visit_map"):
        env.visit_map = np.zeros((env.size, env.size), dtype=np.float32)
        env.visit_map[env.agent_pos[0], env.agent_pos[1]] += 1.0
        
    if not hasattr(env, "action_history"):
        env.action_history = deque([[0.0]*5 for _ in range(8)], maxlen=8)

    obs = []
    
    # 1. Işınlar (Raycasts)
    dirs = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]
    
    # Statik engelleri row, col formatında set yapalım (PPO notebook ile %100 uyumlu)
    static_obs = set()
    for r in range(env.size):
        for c in range(env.size):
            if env.grid[r, c] == 1:
                static_obs.add((r, c))

    # Dinamik engelleri PPO modelinin beklentisine uygun biçimde tanımla
    class PPODynObs:
        def __init__(self, row, col, dr, dc):
            self.x = float(row)
            self.y = float(col)
            self.vx = float(dr)
            self.vy = float(dc)
        @property
        def grid_pos(self):
            return (int(round(self.x)), int(round(self.y)))
        def normalized_velocity(self, max_speed):
            # Ölçeği PPO eğitimindeki sabit 0.5 hızına göre normalize et (en kritik çarpışma önleme hassasiyeti)
            vx_scaled = 0.5 if self.vx > 0 else (-0.5 if self.vx < 0 else 0.0)
            vy_scaled = 0.5 if self.vy > 0 else (-0.5 if self.vy < 0 else 0.0)
            return (vx_scaled, vy_scaled)

    ppo_dyn_obs = []
    for o in env.dynamic_obstacles:
        # DummyObs sınıfından dr (row farkı) ve dc (col farkı) yönlerini doğrudan alıyoruz
        ppo_dyn_obs.append(PPODynObs(o.row, o.col, o.dr, o.dc))

    dyn_positions_dict = {dyn.grid_pos: dyn for dyn in ppo_dyn_obs}

    for dx, dy in dirs:
        hit_data = [1.0, 0.0, 0.0, 0.0]
        for step in range(1, view_radius + 1):
            rx, ry = env.agent_pos[0] + dx*step, env.agent_pos[1] + dy*step
            if rx < 0 or rx >= env.size or ry < 0 or ry >= env.size or (rx, ry) in static_obs:
                hit_data = [step/view_radius, 1.0, 0.0, 0.0]
                break
            if (rx, ry) in dyn_positions_dict:
                dyn = dyn_positions_dict[(rx, ry)]
                nvx, nvy = dyn.normalized_velocity(1.0)
                hit_data = [step/view_radius, 0.5, nvx, nvy]
                break
        obs.extend(hit_data)

    # 2. Hedefe Kalan Mesafe ve Açı (PPO modelinde x=row, y=col koordinat farkıdır)
    delta_x = (env.goal_pos[0] - env.agent_pos[0]) / env.size
    delta_y = (env.goal_pos[1] - env.agent_pos[1]) / env.size
    dist_norm = math.hypot(delta_x, delta_y) / math.sqrt(2)
    angle = math.atan2(delta_y, delta_x)
    obs.extend([delta_x, delta_y, dist_norm, math.sin(angle), math.cos(angle)])

    # 3. Ziyaret Haritası (Agent merkezli 5x5 bölge, doğrudan row/col şeklinde)
    max_vis = max(1.0, np.max(env.visit_map))
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            px, py = env.agent_pos[0]+dx, env.agent_pos[1]+dy
            if 0 <= px < env.size and 0 <= py < env.size:
                obs.append(env.visit_map[px, py] / max_vis)
            else:
                obs.append(1.0)

    # 4. Aksiyon Geçmişi
    for act_arr in env.action_history:
        obs.extend(act_arr)

    return np.array(obs, dtype=np.float32)


@app.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket):
    """
    Spring Boot uyumlu WebSocket simülasyon endpointi.
    Her tick'te ajanın bir adımını hesaplar ve sonucu iletir.
    """
    global env
    await ws.accept()
    print(f"[WS] İstemci bağlandı: {ws.client}")

    # Global env'i kullan — /maps/load ile initialize edilmiş durumda

    from collections import deque as _deque
    pos_history = _deque(maxlen=10)  # Stuck detection için pozisyon geçmişi

    try:
        while True:
            try:
                raw = await ws.receive_text()
            except Exception as recv_err:
                print(f"[WS] Receive hata: {recv_err}")
                break

            try:
                tick = json.loads(raw)
            except json.JSONDecodeError as json_err:
                print(f"[WS] JSON decode hatası: {json_err}")
                await ws.send_json({"error": "Geçersiz JSON"})
                continue

            try:
                # Harita güncelleme (opsiyonel)
                if tick.get("grid") is not None:
                    grid_arr = np.array(tick["grid"], dtype=np.int8)
                    size = grid_arr.shape[0]
                    half = size // 2

                    def api_to_idx(x: int, y: int) -> tuple:
                        return (half - y, x + half)

                    sp = tick.get("agent_pos", {"x": -half, "y": half})
                    gp = tick.get("goal_pos", {"x": half, "y": -half})
                    start = api_to_idx(sp["x"], sp["y"])
                    goal = api_to_idx(gp["x"], gp["y"])

                    env._generate_from_data(grid_arr, start, goal)
                    # Agent'ı başlangıç konumuna reset et
                    env.agent_pos = start
                    env.steps_taken = 0
                    env._prev_dist = float(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
                    env._visited = {start: 1}
                    if IS_PPO:
                        if hasattr(env, "visit_map"): delattr(env, "visit_map")
                        if hasattr(env, "action_history"): delattr(env, "action_history")

                # Hareketli engellerin senkronizasyonu
                if tick.get("dynamic_obstacles") is not None:
                    class DummyObs:
                        def __init__(self, pos, dr=0.0, dc=0.0):
                            self.pos = pos
                            self.row = pos[0]
                            self.col = pos[1]
                            self.dr = dr
                            self.dc = dc
                    
                    # Önceki adımdaki engellerin konum kopyasını al (Geçiş çarpışması denetimi ve yön hesabı için)
                    old_dyn_obs = list(env.dynamic_obstacles) if hasattr(env, "dynamic_obstacles") else []

                    dyn_obs = []
                    half = env.size // 2
                    def api_to_idx(x: int, y: int) -> tuple:
                        return (half - y, x + half)
                        
                    for idx, d_cart in enumerate(tick["dynamic_obstacles"]):
                        d_idx = api_to_idx(d_cart["x"], d_cart["y"])
                        # Önceki adıma göre hareket yönünü (dr, dc) hesapla
                        dr, dc = 0.0, 0.0
                        if idx < len(old_dyn_obs):
                            prev_obs = old_dyn_obs[idx]
                            dr = float(d_idx[0] - prev_obs.row)
                            dc = float(d_idx[1] - prev_obs.col)
                        dyn_obs.append(DummyObs(d_idx, dr, dc))
                    
                    env.prev_dynamic_obstacles = old_dyn_obs  # Gerçek önceki konumlar
                    env.dynamic_obstacles = dyn_obs           # Gerçek yeni konumlar

                # Geçiş çarpışması denetimi için ajanın mevcut konumunu yedekle
                prev_agent_pos = (env.agent_pos[0], env.agent_pos[1])

                # Durum vektörü ve Inference (Model tipine göre)
                if IS_PPO:
                    state = get_ppo_observation(env)
                    # PPO logits'i alıp frontend Q-değerleri olarak gönderelim
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    with torch.no_grad():
                        logits = agent.policy(state_t).squeeze(0).numpy()
                    
                    # PPO: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=STAY
                    # FE:  0=LEFT, 1=RIGHT, 2=UP, 3=DOWN, 4=STAY (Birebir 1-to-1 uyum!)
                    ppo_action = int(np.argmax(logits))
                    action = ppo_action

                    # --- GÜVENLİK KALKANI (COLLISION AVOIDANCE SHIELD) ---
                    # Eğer ajan statik bir duvara veya sınır dışına çarpmak üzereyse
                    # beyninin karar verdiği en yüksek logitli GÜVENLİ alternatife yönlendirilir.
                    DELTA_MAP = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}
                    next_pos = None
                    if action in DELTA_MAP:
                        dr, dc = DELTA_MAP[action]
                        next_pos = (env.agent_pos[0] + dr, env.agent_pos[1] + dc)
                    
                    if next_pos is not None and (
                        not (0 <= next_pos[0] < env.size and 0 <= next_pos[1] < env.size)
                        or env.grid[next_pos[0], next_pos[1]] == 1
                    ):
                        safe_actions = []
                        for a in range(5):
                            if a == 4:
                                safe_actions.append(a)
                            elif a in DELTA_MAP:
                                cdr, cdc = DELTA_MAP[a]
                                c_pos = (env.agent_pos[0] + cdr, env.agent_pos[1] + cdc)
                                if (0 <= c_pos[0] < env.size and 0 <= c_pos[1] < env.size) and env.grid[c_pos[0], c_pos[1]] != 1:
                                    safe_actions.append(a)
                        if safe_actions:
                            action = int(max(safe_actions, key=lambda a: logits[a]))
                            ppo_action = action
                    
                    # Update PPO action history
                    action_oh = [0.0]*5
                    action_oh[ppo_action] = 1.0
                    env.action_history.append(action_oh)
                    
                    # Q-values representation for frontend
                    q_values = [
                        float(logits[0]),  # LEFT
                        float(logits[1]),  # RIGHT
                        float(logits[2]),  # UP
                        float(logits[3]),  # DOWN
                    ]
                    
                    # Simülasyon adımı
                    if action == 4:  # STAY
                        reward = -0.05
                        env.steps_taken += 1
                        done = env.steps_taken >= env.max_steps
                        info = {"reached_goal": False, "steps": env.steps_taken}
                    else:
                        _, reward, done, info = env.step(action)
                    
                    epsilon = 0.0  # PPO deterministik çalışıyor
                    episode = 1
                else:
                    if tick.get("state") is not None:
                        state = np.array(tick["state"], dtype=np.float32)[:agent.state_size]
                    else:
                        state = env._get_state()[:agent.state_size]
                        
                    q_values = agent.get_q_values(state)
                    DELTA_MAP = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}
                    
                    # Salınım tespiti: son 6 adımda ≤2 unique pozisyon
                    oscillating = (
                        len(pos_history) >= 6
                        and len(set(list(pos_history)[-6:])) <= 2
                    )
                    
                    if oscillating:
                        recent = set(list(pos_history)[-4:])
                        escape = [a for a in range(4)
                                  if tuple(np.array(env.agent_pos) + np.array(DELTA_MAP[a])) not in recent]
                        candidates = escape if escape else list(range(4))
                        action = int(max(candidates, key=lambda a: q_values[a]))
                    else:
                        action = int(np.argmax(q_values))
                        
                    _, reward, done, info = env.step(action)
                    epsilon = round(float(agent.epsilon), 4)
                    episode = int(agent.episode_count)

                # ─── GEÇİŞ ÇARPIŞMASI (SWAP COLLISION) KONTROLÜ ───
                # Eğer ajan ve herhangi bir hareketli engel hücre değiştirdiyse (birbirinin içinden geçtiyse)
                # bu durum da çarpışmadır ve simülasyon sonlandırılmalıdır!
                swap_hit = False
                new_agent_pos = (env.agent_pos[0], env.agent_pos[1])
                if not done:
                    for idx, obs in enumerate(env.dynamic_obstacles):
                        if idx < len(env.prev_dynamic_obstacles):
                            prev_obs = env.prev_dynamic_obstacles[idx]
                            prev_obs_pos = (prev_obs.row, prev_obs.col)
                            new_obs_pos = (obs.row, obs.col)
                            
                            # Eğer ajan engelin eski yerine, engel de ajanın eski yerine geldiyse
                            if prev_agent_pos == new_obs_pos and new_agent_pos == prev_obs_pos:
                                swap_hit = True
                                print(f"[COLLISION] Geçiş (Swap) Çarpışması! Ajan: {prev_agent_pos}->{new_agent_pos} | Engel: {prev_obs_pos}->{new_obs_pos}")
                                done = True
                                reward = -50.0
                                info = {"reached_goal": False, "status": "hit_obstacle", "steps": env.steps_taken}
                                break

                # Hareketli engel ajanın üzerine mi geldi? (Pasif veya Geçiş Çarpışması)
                if swap_hit or (env._is_blocked(env.agent_pos[0], env.agent_pos[1]) and env.grid[env.agent_pos[0], env.agent_pos[1]] != 1):
                    # Hareketli engel ajanı ezdi!
                    r, c = env.agent_pos
                    half = env.size // 2
                    await ws.send_json({
                        "action": 0, "action_label": "HIT", "q_values": [0,0,0,0],
                        "reward": -50.0, "done": True, "reached_goal": False, "stuck": False,
                        "agent_pos": {"x": int(c - half), "y": int(half - r)}
                    })
                    pos_history.clear()
                    env.reset()
                    if IS_PPO:
                        if hasattr(env, "visit_map"): delattr(env, "visit_map")
                        if hasattr(env, "action_history"): delattr(env, "action_history")
                    continue

                # Ziyaret haritasını güncelle (PPO)
                if IS_PPO:
                    if not hasattr(env, "visit_map"):
                        env.visit_map = np.zeros((env.size, env.size), dtype=np.float32)
                    env.visit_map[env.agent_pos[0], env.agent_pos[1]] += 1.0

                # Kartezyen koordinata dönüştür (frontend için)
                r, c = env.agent_pos
                half = env.size // 2
                agent_x = int(c - half)
                agent_y = int(half - r)
                pos_history.append((r, c))

                action_labels = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN", 4: "STAY"}
                print(f"[SIM] Step {env.steps_taken}: Cart({agent_x},{agent_y}) | {action_labels.get(action, 'UNKNOWN')} | Q={[round(x, 2) for x in q_values[:4]]}")
                
                response = {
                    "action": int(action),
                    "action_label": action_labels.get(action, "UNKNOWN"),
                    "q_values": [round(float(v), 4) for v in q_values],
                    "reward": round(float(reward), 2),
                    "done": bool(done),
                    "reached_goal": bool(info["reached_goal"]),
                    "stuck": bool(info.get("stuck", False)),
                    "epsilon": epsilon,
                    "episode": episode,
                    "agent_pos": {"x": agent_x, "y": agent_y},
                    "steps": int(info.get("steps", env.steps_taken)),
                }
                await ws.send_json(response)

                if done:
                    pos_history.clear()
                    env.reset()
                    if IS_PPO:
                        if hasattr(env, "visit_map"): delattr(env, "visit_map")
                        if hasattr(env, "action_history"): delattr(env, "action_history")
            except Exception as proc_err:
                print(f"[WS] İşlem hatası: {proc_err}")
                import traceback
                traceback.print_exc()
                # Try to send error response
                try:
                    await ws.send_json({"error": f"İşlem hatası: {str(proc_err)}"})
                except:
                    break

    except WebSocketDisconnect:
        print(f"[WS] İstemci ayrıldı: {ws.client}")


# ─── Başlatma ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Production-ready: reload causes state management issues
        log_level="info",
    )
