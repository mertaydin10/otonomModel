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
    os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pth"
)
DEFAULT_SIZE = 15

env = GridEnvironment(size=DEFAULT_SIZE, random_maps=True)
agent = DQLAgent(state_size=env.state_size, action_size=env.action_size)
agent.load(MODEL_PATH)

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


class MapPayload(BaseModel):
    map_name: str
    grid_size: Dict[str, int]
    start_pos: Dict[str, int]
    target_pos: Dict[str, int]
    obstacles: Dict[str, list]


# ─── Endpointler ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
        "device": str(agent.device),
        "episode": agent.episode_count,
        "state_size": env.state_size,
    }


@app.post("/maps/load")
async def load_map(payload: MapPayload):
    """Spring Boot'tan gelen GameMapDTO'yu yükle."""
    try:
        env.load_from_api_payload(payload.model_dump())
        env.reset()
        return {
            "status": "ok",
            "map_name": payload.map_name,
            "grid_size": payload.grid_size,
            "start": {"row": env.start_pos[0], "col": env.start_pos[1]},
            "goal": {"row": env.goal_pos[0], "col": env.goal_pos[1]},
        }
    except Exception as exc:
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
    await ws.accept()
    print(f"[WS] İstemci bağlandı: {ws.client}")

    sim_env = GridEnvironment(size=DEFAULT_SIZE, random_maps=True)
    sim_env.reset()

    try:
        while True:
            raw = await ws.receive_text()
            try:
                tick = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"error": "Geçersiz JSON"})
                continue

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

                sim_env.size = size
                sim_env._generate_from_data(grid_arr, start, goal)
                sim_env.steps_taken = 0

            # Durum vektörü
            if tick.get("state") is not None:
                state = np.array(tick["state"], dtype=np.float32)
            else:
                state = sim_env._get_state()

            # Inference
            action = agent.select_action(state, training=False)
            q_values = agent.get_q_values(state)

            # Simülasyonu ilerlet
            _, reward, done, info = sim_env.step(action)

            # Kartezyen koordinata dönüştür (frontend için)
            r, c = sim_env.agent_pos
            half = sim_env.size // 2
            response = {
                "action": action,
                "action_label": ACTION_LABELS[action],
                "q_values": [round(float(v), 4) for v in q_values],
                "reward": round(float(reward), 2),
                "done": done,
                "reached_goal": info["reached_goal"],
                "epsilon": round(agent.epsilon, 4),
                "episode": agent.episode_count,
                "agent_pos": {"x": c - half, "y": half - r},
                "steps": info["steps"],
            }
            await ws.send_json(response)

            if done:
                sim_env.reset()

    except WebSocketDisconnect:
        print(f"[WS] İstemci ayrıldı: {ws.client}")


# ─── Başlatma ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info",
    )
