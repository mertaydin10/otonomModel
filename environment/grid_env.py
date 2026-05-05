"""
environment/grid_env.py — Grid World ortamı

Aksiyon uzayı (Spring Boot SimulationResponseDTO ile uyumlu):
  0 = LEFT  ←  (col − 1)
  1 = RIGHT →  (col + 1)
  2 = UP    ↑  (row − 1)
  3 = DOWN  ↓  (row + 1)

Durum vektörü (12 eleman — Spring Boot Normalizer ile uyumlu):
  [0]  agent_col / (size-1)          — ajan x normalize
  [1]  agent_row / (size-1)          — ajan y normalize
  [2]  goal_col  / (size-1)          — hedef x normalize
  [3]  goal_row  / (size-1)          — hedef y normalize
  [4]  sensor_left  (0/1 anlık engel)
  [5]  sensor_right (0/1 anlık engel)
  [6]  sensor_up    (0/1 anlık engel)
  [7]  sensor_down  (0/1 anlık engel)
  [8]  dist_left  / (size-1)         — sol engele normalize mesafe
  [9]  dist_right / (size-1)         — sağ engele normalize mesafe
  [10] dist_up    / (size-1)         — üst engele normalize mesafe
  [11] dist_down  / (size-1)         — alt engele normalize mesafe
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional

# ─── Aksiyon sabitleri (Spring Boot uyumlu) ──────────────────────────────────
ACTION_LEFT  = 0   # ←
ACTION_RIGHT = 1   # →
ACTION_UP    = 2   # ↑
ACTION_DOWN  = 3   # ↓

DELTA = {
    ACTION_LEFT:  ( 0, -1),
    ACTION_RIGHT: ( 0, +1),
    ACTION_UP:    (-1,  0),
    ACTION_DOWN:  (+1,  0),
}

ACTION_LABELS = {
    ACTION_LEFT:  "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_UP:    "UP",
    ACTION_DOWN:  "DOWN",
}


class GridEnvironment:
    """
    Grid World ortamı — DQL ajan eğitimi için.

    Özellikler:
    - Çözülebilirlik garantili rastgele harita üretimi (BFS doğrulamalı)
    - Spring Boot Normalizer ile uyumlu 12-boyutlu durum vektörü
    - Doğru ödül şekillendirmesi (önceki konuma göre mesafe farkı)
    - GameMapDTO formatından harita yükleme
    - Bölüm başına maksimum adım sınırı
    """

    def __init__(
        self,
        size: int = 15,
        obstacle_ratio: float = 0.15,
        max_steps: int = 500,
        random_maps: bool = True,
        min_path_length: int = 0,
    ):
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = max_steps
        self.random_maps = random_maps
        self.min_path_length = min_path_length

        self.grid = np.zeros((size, size), dtype=np.int8)
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self._prev_dist: float = 0.0

        self.state_size = 12   # Spring Boot Normalizer ile uyumlu
        self.action_size = 4

        if self.random_maps:
            self._generate_random_map()
        else:
            self._reset_episode()

    # ─── Harita Üretimi ───────────────────────────────────────────────────────

    def _is_solvable(self, grid: np.ndarray, start: tuple, goal: tuple) -> bool:
        """BFS ile başlangıç → hedef yolunun var olup olmadığını doğrular."""
        return self._bfs_path_length(grid, start, goal) >= 0

    def _bfs_path_length(
        self, grid: np.ndarray, start: tuple, goal: tuple
    ) -> int:
        """
        BFS ile en kısa yol adım sayısını döndürür.
        Ulaşılamazsa veya geçersiz konumsa -1.
        """
        h, w = grid.shape
        sr, sc = int(start[0]), int(start[1])
        gr, gc = int(goal[0]), int(goal[1])
        if not (0 <= sr < h and 0 <= sc < w and 0 <= gr < h and 0 <= gc < w):
            return -1
        if grid[sr, sc] == 1 or grid[gr, gc] == 1:
            return -1
        if (sr, sc) == (gr, gc):
            return 0
        visited = {(sr, sc)}
        q: deque = deque([(sr, sc, 0)])
        while q:
            row, col, dist = q.popleft()
            for dr, dc in DELTA.values():
                nr, nc = row + dr, col + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if grid[nr, nc] != 0 or (nr, nc) in visited:
                    continue
                if (nr, nc) == (gr, gc):
                    return dist + 1
                visited.add((nr, nc))
                q.append((nr, nc, dist + 1))
        return -1

    def _generate_random_map(self) -> None:
        """
        BFS kontrolü + minimum yol uzunluğu koşulunu geçene kadar
        rastgele harita dener.

        min_path_length > 0 ise BFS yolu en az bu kadar adım olmalı
        (kısa/trivial haritalar elenir → daha zor senaryolar).
        """
        while True:
            grid = np.zeros((self.size, self.size), dtype=np.int8)
            n_obs = int(self.size * self.size * self.obstacle_ratio)
            indices = np.random.choice(self.size * self.size, n_obs, replace=False)
            coords = np.unravel_index(indices, (self.size, self.size))
            grid[coords] = 1

            free = np.argwhere(grid == 0)
            if len(free) < 2:
                continue

            si, gi = np.random.choice(len(free), 2, replace=False)
            start = tuple(free[si])
            goal  = tuple(free[gi])

            path_len = self._bfs_path_length(grid, start, goal)
            if path_len < 0:
                continue                          # çözümsüz
            if self.min_path_length > 0 and path_len < self.min_path_length:
                continue                          # yol çok kısa → atla

            self.grid      = grid
            self.start_pos = start
            self.goal_pos  = goal
            break

    def _generate_from_data(
        self, grid: np.ndarray, start: tuple, goal: tuple
    ) -> None:
        """
        Dışarıdan verilmiş grid matrisi ve konum bilgisiyle haritayı ayarlar.
        Harita geçersizse (çözümsüz) random haritaya geri döner.
        """
        if self._is_solvable(grid, start, goal):
            self.grid = grid.copy()
            self.start_pos = tuple(start)
            self.goal_pos = tuple(goal)
            self.size = grid.shape[0]
        else:
            self._generate_random_map()

    def load_from_api_payload(self, payload: dict) -> None:
        """
        Spring Boot'tan gelen GameMapDTO formatındaki sözlüğü yükler.

        payload örneği:
        {
          "map_name":   "harita1",
          "grid_size":  {"x": 15, "y": 15},
          "start_pos":  {"x": -5, "y":  0},
          "target_pos": {"x":  5, "y":  0},
          "obstacles":  {
            "static":  [{"x": 0, "y": 1, "w": 1, "h": 1}],
            "dynamic": []
          }
        }
        """
        size = payload["grid_size"]["x"]
        half = size // 2

        def cart_to_idx(x, y):
            """Kartezyen (x,y) → (row, col)"""
            return (half - y, x + half)

        sp = payload["start_pos"]
        tp = payload.get("target_pos") or payload.get("goal_pos", {})
        start = cart_to_idx(sp["x"], sp["y"])
        goal = cart_to_idx(tp["x"], tp["y"])

        grid = np.zeros((size, size), dtype=np.int8)
        for obs in payload.get("obstacles", {}).get("static", []):
            r, c = cart_to_idx(obs["x"], obs["y"])
            if 0 <= r < size and 0 <= c < size:
                grid[r, c] = 1

        self._generate_from_data(grid, start, goal)

    # ─── Ortam API ────────────────────────────────────────────────────────────

    def reset(
        self,
        grid_data: Optional[np.ndarray] = None,
        start: Optional[tuple] = None,
        goal: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Ortamı sıfırlar ve başlangıç durum vektörünü döndürür.

        Args:
            grid_data: Verilirse bu grid kullanılır (None ise mevcut/random).
            start:     Başlangıç konumu (None ise mevcut).
            goal:      Hedef konumu (None ise mevcut).
        """
        if grid_data is not None and start is not None and goal is not None:
            self._generate_from_data(grid_data, start, goal)
        elif self.random_maps:
            self._generate_random_map()

        self._reset_episode()
        return self._get_state()

    def _reset_episode(self) -> None:
        """Episode değişkenlerini başlangıca al."""
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self._prev_dist = float(
            abs(self.agent_pos[0] - self.goal_pos[0])
            + abs(self.agent_pos[1] - self.goal_pos[1])
        )
        self._visited: dict = {self.start_pos: 1}  # hücre → ziyaret sayısı

    def step(self, action: int) -> tuple:
        """
        Ajan bir adım atar.

        Returns:
            (next_state, reward, done, info)
        """
        if action not in DELTA:
            raise ValueError(f"Geçersiz aksiyon: {action}")

        self.steps_taken += 1
        dr, dc = DELTA[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc

        info: dict = {"steps": self.steps_taken, "reached_goal": False}

        # Sınır dışı
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            info["status"] = "out_of_bounds"
            return self._get_state(), -10.0, True, info

        # Engel çarpması
        if self.grid[nr, nc] == 1:
            info["status"] = "hit_obstacle"
            return self._get_state(), -50.0, True, info

        # Konumu güncelle
        self.agent_pos = (nr, nc)
        visit_count = self._visited.get(self.agent_pos, 0) + 1
        self._visited[self.agent_pos] = visit_count

        # Hedefe ulaşma
        if self.agent_pos == self.goal_pos:
            info["status"] = "goal_reached"
            info["reached_goal"] = True
            self._prev_dist = 0.0
            return self._get_state(), +100.0, True, info

        # Maksimum adım
        if self.steps_taken >= self.max_steps:
            info["status"] = "max_steps_reached"
            return self._get_state(), -1.0, True, info

        # Mesafe bazlı ödül şekillendirmesi
        curr_dist = float(
            abs(self.agent_pos[0] - self.goal_pos[0])
            + abs(self.agent_pos[1] - self.goal_pos[1])
        )
        if curr_dist < self._prev_dist:
            reward = +2.0        # +1.0 → +2.0: hedefe yaklaşma sinyali güçlendirildi
        else:
            reward = -0.5
        reward -= 0.1            # adım cezası (gereksiz dolaşmayı önler)

        self._prev_dist = curr_dist

        info["status"] = "ok"
        return self._get_state(), reward, False, info

    # ─── Durum Vektörü (12 eleman) ───────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        12-boyutlu durum vektörü üretir (Spring Boot Normalizer ile uyumlu).

        [0..3]  Normalize pozisyonlar (agent_col, agent_row, goal_col, goal_row)
        [4..7]  Anlık engel sensörleri (LEFT, RIGHT, UP, DOWN)
        [8..11] Yön bazlı engel mesafeleri normalize (LEFT, RIGHT, UP, DOWN)
        """
        s = max(self.size - 1, 1)
        row, col = self.agent_pos
        grow, gcol = self.goal_pos

        imm = np.zeros(4, dtype=np.float32)
        dist = np.zeros(4, dtype=np.float32)

        for act in range(4):      # 0=LEFT,1=RIGHT,2=UP,3=DOWN
            dr, dc = DELTA[act]
            nr, nc = row + dr, col + dc

            # Anlık sensör
            if not (0 <= nr < self.size and 0 <= nc < self.size) or self.grid[nr, nc] == 1:
                imm[act] = 1.0

            # Mesafe sensörü (ışın atışı)
            r, c = row, col
            d = 0
            while True:
                r, c = r + dr, c + dc
                if not (0 <= r < self.size and 0 <= c < self.size):
                    break
                d += 1
                if self.grid[r, c] == 1:
                    break
            dist[act] = d / s

        state = np.array([
            col / s, row / s,           # ajan x, y
            gcol / s, grow / s,         # hedef x, y
            imm[0], imm[1], imm[2], imm[3],
            dist[0], dist[1], dist[2], dist[3],
        ], dtype=np.float32)

        return state

    # ─── Debug ───────────────────────────────────────────────────────────────

    def render(self) -> None:
        """Grid'i terminalde görselleştirir."""
        symbols = np.full((self.size, self.size), " · ")
        symbols[self.grid == 1] = "███"
        symbols[self.start_pos] = " S "
        symbols[self.goal_pos]  = " G "
        if self.agent_pos not in (self.start_pos, self.goal_pos):
            symbols[self.agent_pos] = " A "
        print("\n".join("".join(r) for r in symbols))
        print(f"Adım: {self.steps_taken}  Pos: {self.agent_pos}  Mesafe: {int(self._prev_dist)}")
