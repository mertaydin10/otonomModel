"""
environment/grid_env.py — Grid World ortamı (v3 — hareketli engelli)

Aksiyon uzayı (Spring Boot SimulationResponseDTO ile uyumlu):
  0 = LEFT  ←  (col − 1)
  1 = RIGHT →  (col + 1)
  2 = UP    ↑  (row − 1)
  3 = DOWN  ↓  (row + 1)

Durum vektörü (16 eleman — v3):
  [0]  agent_col / (size-1)          — ajan x normalize
  [1]  agent_row / (size-1)          — ajan y normalize
  [2]  goal_col  / (size-1)          — hedef x normalize
  [3]  goal_row  / (size-1)          — hedef y normalize
  [4]  sensor_left  (0/1 anlık engel — statik+dinamik)
  [5]  sensor_right (0/1 anlık engel — statik+dinamik)
  [6]  sensor_up    (0/1 anlık engel — statik+dinamik)
  [7]  sensor_down  (0/1 anlık engel — statik+dinamik)
  [8]  dist_left  / (size-1)         — sol engele normalize mesafe
  [9]  dist_right / (size-1)         — sağ engele normalize mesafe
  [10] dist_up    / (size-1)         — üst engele normalize mesafe
  [11] dist_down  / (size-1)         — alt engele normalize mesafe
  [12] nearest_dyn_dx / (size-1)     — en yakın dinamik engel relative x
  [13] nearest_dyn_dy / (size-1)     — en yakın dinamik engel relative y
  [14] nearest_dyn_move_dx           — en yakın dyn. engel hareket yönü x (-1/0/+1)
  [15] nearest_dyn_move_dy           — en yakın dyn. engel hareket yönü y (-1/0/+1)
"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional, List

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


# ─── Dinamik Engel ──────────────────────────────────────────────────────────

class DynamicObstacle:
    """
    Hareketli engel — patrol (devriye) modunda hareket eder.
    Duvara veya statik engele çarpınca yön tersler (bounce).
    """
    __slots__ = ("row", "col", "dr", "dc", "size")

    def __init__(self, row: int, col: int, dr: int, dc: int, size: int):
        self.row = row
        self.col = col
        self.dr = dr      # satır hareket yönü (-1, 0, +1)
        self.dc = dc      # sütun hareket yönü (-1, 0, +1)
        self.size = size

    @property
    def pos(self) -> tuple:
        return (self.row, self.col)

    def move(self, static_grid: np.ndarray, occupied: set) -> None:
        """
        Bir adım ilerle. Eğer hedef hücre duvar, statik engel,
        sınır dışı veya başka bir dinamik engel tarafından tutuluyorsa
        yönü tersle (bounce) ve bir adım daha dene.
        İki denemede de hareket edemezse yerinde kal.
        """
        for attempt in range(2):
            nr = self.row + self.dr
            nc = self.col + self.dc
            if (0 <= nr < self.size and 0 <= nc < self.size
                    and static_grid[nr, nc] == 0
                    and (nr, nc) not in occupied):
                occupied.discard((self.row, self.col))
                self.row = nr
                self.col = nc
                occupied.add((self.row, self.col))
                return
            # Bounce — yönü tersle
            self.dr = -self.dr
            self.dc = -self.dc

    def __repr__(self) -> str:
        return f"DynObs({self.row},{self.col} dir=({self.dr},{self.dc}))"


class GridEnvironment:
    """
    Grid World ortamı — DQL ajan eğitimi için (v3 — hareketli engelli).

    Özellikler:
    - Çözülebilirlik garantili rastgele harita üretimi (BFS doğrulamalı)
    - Spring Boot Normalizer ile uyumlu durum vektörü
    - Doğru ödül şekillendirmesi (önceki konuma göre mesafe farkı)
    - Hareketli engeller (patrol/bounce)
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
        dynamic_obstacle_count: int = 0,
        dynamic_move_interval: int = 1,
    ):
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.max_steps = max_steps
        self.random_maps = random_maps
        self.min_path_length = min_path_length

        # Dinamik engel parametreleri
        self.dynamic_obstacle_count = dynamic_obstacle_count
        self.dynamic_move_interval = dynamic_move_interval  # kaç step'te bir hareket

        self.grid = np.zeros((size, size), dtype=np.int8)
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.agent_pos = self.start_pos
        self.steps_taken = 0
        self._prev_dist: float = 0.0

        # Dinamik engel listesi
        self.dynamic_obstacles: List[DynamicObstacle] = []

        # State boyutu: 12 (statik) + 4 (dinamik) = 16
        # dynamic_obstacle_count == 0 olduğunda bile 16 tutuyoruz
        # böylece aynı ağ her iki modda da çalışabilir
        self.state_size = 16
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

    # ─── Dinamik Engel Yönetimi ──────────────────────────────────────────────

    def _spawn_dynamic_obstacles(self) -> None:
        """
        Rastgele pozisyon ve yönlerle dinamik engeller oluşturur.
        Start, goal ve statik engel olmayan hücrelere yerleştirilir.
        """
        self.dynamic_obstacles.clear()
        if self.dynamic_obstacle_count <= 0:
            return

        # Kullanılabilir hücreleri bul
        forbidden = {self.start_pos, self.goal_pos}
        free_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] == 0 and (r, c) not in forbidden:
                    free_cells.append((r, c))

        if len(free_cells) == 0:
            return

        # Rastgele konumlar seç
        n = min(self.dynamic_obstacle_count, len(free_cells))
        chosen_indices = np.random.choice(len(free_cells), n, replace=False)

        # Yön seçenekleri — sadece yatay veya dikey (çapraz hareket yok)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for idx in chosen_indices:
            r, c = free_cells[idx]
            dr, dc = directions[np.random.randint(len(directions))]
            obs = DynamicObstacle(r, c, dr, dc, self.size)
            self.dynamic_obstacles.append(obs)

    def _move_dynamic_obstacles(self) -> None:
        """Tüm dinamik engelleri bir adım hareket ettirir."""
        if not self.dynamic_obstacles:
            return

        # Mevcut dinamik engellerin pozisyon kümesi
        occupied = {obs.pos for obs in self.dynamic_obstacles}

        for obs in self.dynamic_obstacles:
            obs.move(self.grid, occupied)

    def _is_dynamic_obstacle(self, row: int, col: int) -> bool:
        """Verilen hücrede dinamik engel var mı?"""
        for obs in self.dynamic_obstacles:
            if obs.row == row and obs.col == col:
                return True
        return False

    def _is_blocked(self, row: int, col: int) -> bool:
        """Verilen hücre statik veya dinamik engelle bloke mu?"""
        if self.grid[row, col] == 1:
            return True
        return self._is_dynamic_obstacle(row, col)

    def _get_dynamic_positions(self) -> set:
        """Dinamik engel pozisyonları kümesi."""
        return {obs.pos for obs in self.dynamic_obstacles}

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

        # Dinamik engelleri yeniden oluştur
        self._spawn_dynamic_obstacles()

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

        # Statik engel çarpması
        if self.grid[nr, nc] == 1:
            info["status"] = "hit_obstacle"
            return self._get_state(), -50.0, True, info

        # Dinamik engel çarpması (ajanın gittiği hücre)
        if self._is_dynamic_obstacle(nr, nc):
            info["status"] = "hit_dynamic_obstacle"
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

        # ── Dinamik engelleri hareket ettir (ajan adımından SONRA) ──────────
        if (self.dynamic_obstacles
                and self.steps_taken % self.dynamic_move_interval == 0):
            self._move_dynamic_obstacles()

            # Hareket ettikten sonra dinamik engel ajanın üstüne geldiyse
            if self._is_dynamic_obstacle(*self.agent_pos):
                info["status"] = "dynamic_obstacle_hit_agent"
                return self._get_state(), -50.0, True, info

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
            reward = +2.0        # hedefe yaklaşma sinyali
        else:
            reward = -0.5
        reward -= 0.1            # adım cezası (gereksiz dolaşmayı önler)

        # Dinamik engele yakınlık bonusu/cezası
        if self.dynamic_obstacles:
            min_dyn_dist = self._nearest_dynamic_dist()
            if min_dyn_dist <= 1:
                reward -= 1.0    # çok yakın — tehlike cezası
            elif min_dyn_dist <= 2:
                reward -= 0.3    # yaklaşıyor — hafif ceza

        self._prev_dist = curr_dist

        info["status"] = "ok"
        return self._get_state(), reward, False, info

    # ─── Durum Vektörü (16 eleman) ───────────────────────────────────────────

    def _nearest_dynamic_dist(self) -> float:
        """Ajanın en yakın dinamik engele Manhattan mesafesi."""
        if not self.dynamic_obstacles:
            return float(self.size * 2)  # çok uzak
        row, col = self.agent_pos
        min_dist = float("inf")
        for obs in self.dynamic_obstacles:
            d = abs(obs.row - row) + abs(obs.col - col)
            if d < min_dist:
                min_dist = d
        return min_dist

    def _get_state(self) -> np.ndarray:
        """
        16-boyutlu durum vektörü üretir (v3).

        [0..3]  Normalize pozisyonlar (agent_col, agent_row, goal_col, goal_row)
        [4..7]  Anlık engel sensörleri (LEFT, RIGHT, UP, DOWN) — statik + dinamik
        [8..11] Yön bazlı engel mesafeleri normalize (LEFT, RIGHT, UP, DOWN)
        [12..15] En yakın dinamik engel bilgisi
        """
        s = max(self.size - 1, 1)
        row, col = self.agent_pos
        grow, gcol = self.goal_pos

        imm = np.zeros(4, dtype=np.float32)
        dist = np.zeros(4, dtype=np.float32)

        # Dinamik engel pozisyonlarını al
        dyn_positions = self._get_dynamic_positions()

        for act in range(4):      # 0=LEFT,1=RIGHT,2=UP,3=DOWN
            dr, dc = DELTA[act]
            nr, nc = row + dr, col + dc

            # Anlık sensör — statik + dinamik engeller
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                imm[act] = 1.0
            elif self.grid[nr, nc] == 1 or (nr, nc) in dyn_positions:
                imm[act] = 1.0

            # Mesafe sensörü (ışın atışı) — statik + dinamik
            r, c = row, col
            d = 0
            while True:
                r, c = r + dr, c + dc
                if not (0 <= r < self.size and 0 <= c < self.size):
                    break
                d += 1
                if self.grid[r, c] == 1 or (r, c) in dyn_positions:
                    break
            dist[act] = d / s

        # Dinamik engel bilgisi
        nearest_dx = 0.0
        nearest_dy = 0.0
        nearest_move_dx = 0.0
        nearest_move_dy = 0.0

        if self.dynamic_obstacles:
            # En yakın dinamik engeli bul
            min_dist = float("inf")
            nearest_obs = None
            for obs in self.dynamic_obstacles:
                d = abs(obs.row - row) + abs(obs.col - col)
                if d < min_dist:
                    min_dist = d
                    nearest_obs = obs
            if nearest_obs is not None:
                nearest_dx = (nearest_obs.col - col) / s
                nearest_dy = (nearest_obs.row - row) / s
                nearest_move_dx = float(nearest_obs.dc)
                nearest_move_dy = float(nearest_obs.dr)

        state = np.array([
            col / s, row / s,           # ajan x, y
            gcol / s, grow / s,         # hedef x, y
            imm[0], imm[1], imm[2], imm[3],
            dist[0], dist[1], dist[2], dist[3],
            nearest_dx, nearest_dy,     # en yakın dinamik engel göreceli pozisyon
            nearest_move_dx, nearest_move_dy,  # en yakın dinamik engel hareket yönü
        ], dtype=np.float32)

        return state

    # ─── Debug ───────────────────────────────────────────────────────────────

    def render(self) -> None:
        """Grid'i terminalde görselleştirir."""
        symbols = np.full((self.size, self.size), " · ")
        symbols[self.grid == 1] = "███"
        # Dinamik engeller
        for obs in self.dynamic_obstacles:
            symbols[obs.row, obs.col] = " ◆ "
        symbols[self.start_pos] = " S "
        symbols[self.goal_pos]  = " G "
        if self.agent_pos not in (self.start_pos, self.goal_pos):
            symbols[self.agent_pos] = " A "
        print("\n".join("".join(r) for r in symbols))
        print(f"Adım: {self.steps_taken}  Pos: {self.agent_pos}  Mesafe: {int(self._prev_dist)}"
              f"  DynObs: {len(self.dynamic_obstacles)}")
