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

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "best_model_v2.pth"
)
DEFAULT_SIZE = 15

env = GridEnvironment(size=DEFAULT_SIZE, random_maps=True)
# Model config: state_size=12, hidden_size=128 (curriculum training)
agent = DQLAgent(
    state_size=env.state_size,
    action_size=env.action_size,
    hidden_size=128  # Match best_model_curriculum_final.pth
)
try:
    agent.load(MODEL_PATH)
except Exception as _load_err:
    print(f"[WARN] Model yüklenemedi (boyut uyuşmazlığı?), yeni ağırlıklarla başlanıyor: {_load_err}")

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
        env = GridEnvironment(size=size, random_maps=False)

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

                # Durum vektörü
                if tick.get("state") is not None:
                    state = np.array(tick["state"], dtype=np.float32)
                else:
                    state = env._get_state()

                # Inference
                q_values = agent.get_q_values(state)
                DELTA_MAP = {0: (0,-1), 1: (0,1), 2: (-1,0), 3: (1,0)}

                def is_safe(a):
                    dr, dc = DELTA_MAP[a]
                    nr, nc = env.agent_pos[0]+dr, env.agent_pos[1]+dc
                    return (0 <= nr < env.size and 0 <= nc < env.size
                            and env.grid[nr, nc] != 1)

                # Duvar/engele çarpmayan aksiyonlar
                safe_actions = [a for a in range(4) if is_safe(a)]

                # Salınım tespiti: son 6 adımda ≤2 unique pozisyon
                oscillating = (
                    len(pos_history) >= 6
                    and len(set(list(pos_history)[-6:])) <= 2
                )

                if oscillating and safe_actions:
                    # Son 4 adımda ziyaret edilmemiş güvenli aksiyonlar tercih edilir
                    recent = set(list(pos_history)[-4:])
                    escape = [a for a in safe_actions
                              if tuple(np.array(env.agent_pos) + np.array(DELTA_MAP[a])) not in recent]
                    candidates = escape if escape else safe_actions
                    action = max(candidates, key=lambda a: q_values[a])
                    print(f"[WS] ESCAPE: osc={oscillating}, action={action}")
                elif safe_actions:
                    # Normal: güvenli aksiyonlar arasında en yüksek Q
                    action = max(safe_actions, key=lambda a: q_values[a])
                else:
                    # Her yön engelli/duvar — episode bitir
                    action = int(np.argmax(q_values))

                # Simülasyonu ilerlet
                _, reward, done, info = env.step(action)

                # Kartezyen koordinata dönüştür (frontend için)
                r, c = env.agent_pos
                half = env.size // 2
                agent_x = int(c - half)
                agent_y = int(half - r)
                pos_history.append((r, c))

                action_labels = ["LEFT", "RIGHT", "UP", "DOWN"]
                print(f"[SIM] Step {env.steps_taken}: Cart({agent_x},{agent_y}) | {action_labels[action]} | Q=[{q_values[0]:.2f},{q_values[1]:.2f},{q_values[2]:.2f},{q_values[3]:.2f}] | osc={oscillating}")
                response = {
                    "action": int(action),
                    "action_label": str(ACTION_LABELS[action]),
                    "q_values": [round(float(v), 4) for v in q_values],
                    "reward": round(float(reward), 2),
                    "done": bool(done),
                    "reached_goal": bool(info["reached_goal"]),
                    "stuck": bool(info.get("stuck", False)),
                    "epsilon": round(float(agent.epsilon), 4),
                    "episode": int(agent.episode_count),
                    "agent_pos": {"x": agent_x, "y": agent_y},
                    "steps": int(info["steps"]),
                }
                await ws.send_json(response)

                if done:
                    pos_history.clear()
                    env.reset()
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
