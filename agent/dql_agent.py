"""
dql_agent.py — Double Dueling DQN Agent
eps-greedy politika, experience replay ve hedef ağ güncellemesi içerir.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from typing import Optional, Tuple, Dict, Any

from .q_network    import QNetwork
from .replay_buffer import ReplayBuffer


class DQLAgent:
    """
    Double Dueling Deep Q-Learning Agent.

    Özellikler:
    - Double DQN: hedef değerleri online ağ seçer, hedef ağ hesaplar
    - Dueling Network: Q-Network içinde value ve advantage stream'leri
    - eps-greedy keşif stratejisi (üstel azalma)
    - Periyodik hedef ağ güncellemesi
    - Model kaydetme / yükleme
    """

    def __init__(
        self,
        state_size:     int   = 12,
        action_size:    int   = 4,
        hidden_size:    int   = 64,
        learning_rate:  float = 0.001,
        gamma:          float = 0.95,
        epsilon_start:  float = 1.0,
        epsilon_min:    float = 0.01,
        epsilon_decay:  float = 0.995,
        batch_size:     int   = 64,
        buffer_capacity:int   = 10_000,
        target_update:  int   = 10,
        device:         Optional[str] = None,
    ):
        """
        Args:
            state_size:      Durum vektörü boyutu
            action_size:     Aksiyon sayısı
            hidden_size:     Gizli katman nöron sayısı
            learning_rate:   Optimizer öğrenme hızı
            gamma:           İndirim faktörü (gelecek ödül ağırlığı)
            epsilon_start:   Başlangıç keşif oranı
            epsilon_min:     Minimum keşif oranı
            epsilon_decay:   Her episode'da epsilon çarpanı
            batch_size:      Mini-batch boyutu
            buffer_capacity: Replay buffer kapasitesi
            target_update:   Hedef ağ güncelleme sıklığı (episode)
            device:          'cpu', 'cuda', 'mps' veya None (otomatik)
        """
        # Cihaz seçimi (MPS = Apple Silicon GPU)
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.state_size  = state_size
        self.action_size = action_size
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.target_update = target_update

        # eps-greedy parametreleri
        self.epsilon       = epsilon_start
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Ağlar
        self.q_network     = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer & loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn   = nn.SmoothL1Loss()  # Huber loss — daha kararlı eğitim

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # İstatistikler
        self.total_steps    = 0
        self.episode_count  = 0
        self.training_stats: list = []

    # ─── Aksiyon Seçimi ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        eps-greedy aksiyon seçimi.

        Args:
            state:    Durum vektörü
            training: True ise epsilon uygulanır, False ise tamamen açgözlü

        Returns:
            action: 0-3 arası aksiyon ID
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)  # Keşif

        # state: shape (state_size,) — unsqueeze ile (1, state_size) yapılır
        state_tensor = torch.FloatTensor(np.asarray(state).flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).item())  # Sömürü

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Bir durum için Q değerlerini döndür (API için)."""
        state_tensor = torch.FloatTensor(np.asarray(state).flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]

    # ─── Deneyim Ekleme ──────────────────────────────────────────────────────

    def remember(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Deneyimi replay buffer'a ekle."""
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

    # ─── Eğitim Adımı ────────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Bir mini-batch ile Q-Network'ü güncelle.

        Returns:
            loss: Float loss değeri veya None (yeterli deneyim yoksa)
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Tensor'lara çevir — states shape: (batch, state_size)
        states_t      = torch.FloatTensor(np.array(states).reshape(self.batch_size, -1)).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states).reshape(self.batch_size, -1)).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Mevcut Q değerleri
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1))

        # Double DQN hedef hesaplama:
        # 1. Online ağ ile en iyi aksiyonu seç
        # 2. Hedef ağ ile bu aksiyonun Q değerini hesapla
        with torch.no_grad():
            next_actions   = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            next_q         = self.target_network(next_states_t).gather(1, next_actions)
            target_q       = rewards_t.unsqueeze(1) + self.gamma * next_q * (1 - dones_t.unsqueeze(1))

        # Loss hesapla ve geri yayılım
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — kararlılık için
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Her episode sonunda epsilon'u azalt."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    def update_target_network(self) -> None:
        """Hedef ağı online ağın ağırlıklarıyla güncelle."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ─── Model Kaydetme / Yükleme ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Modeli kaydet.

        Kaydedilen dosya içeriği:
        - q_network state_dict
        - Hiperparametreler
        - Eğitim istatistikleri
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "q_network_state":     self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state":     self.optimizer.state_dict(),
            "epsilon":             self.epsilon,
            "total_steps":         self.total_steps,
            "episode_count":       self.episode_count,
            "config": {
                "state_size":  self.state_size,
                "action_size": self.action_size,
                "hidden_size": self.q_network.hidden_size,
                "gamma":       self.gamma,
            },
        }
        torch.save(checkpoint, path)
        print(f"[DQL] Model kaydedildi: {path}")

    def load(self, path: str) -> bool:
        """
        Kaydedilmiş modeli yükle.

        Returns:
            True: Başarılı, False: Dosya bulunamadı
        """
        if not os.path.exists(path):
            print(f"[DQL] Model bulunamadı: {path}")
            return False

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon     = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        self.episode_count = checkpoint["episode_count"]
        print(f"[DQL] Model yuklendi: {path} (Episode: {self.episode_count}, eps: {self.epsilon:.3f})")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Mevcut eğitim istatistiklerini döndür."""
        return {
            "episode":    self.episode_count,
            "epsilon":    round(self.epsilon, 4),
            "total_steps": self.total_steps,
            "buffer_size": len(self.replay_buffer),
            "device":     str(self.device),
        }
