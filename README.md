# Otonom Sürüş — Python AI Servisi

**Dueling Double DQN** ile eğitilmiş, Spring Boot backend ve React frontend ile entegre çalışan otonom araç simülasyonu.

---

## Mimari

```
otonomFront  (React)          otonomBack (Spring Boot)       otonomModel (Python AI)
─────────────────────         ──────────────────────────     ────────────────────────
Grid World UI            ←→   REST /maps/save            →   model.onnx  (ONNX çıkarım)
WebSocket Client         ←→   WS /ws/simulate            →   FastAPI /ws/simulate
```

---

## Klasör Yapısı

```
otonomModel/
├── environment/
│   ├── __init__.py
│   └── grid_env.py          # Grid World ortamı — Spring Boot uyumlu
├── agent/
│   ├── __init__.py
│   ├── q_network.py         # Dueling DQN sinir ağı
│   ├── replay_buffer.py     # Experience Replay (Uniform + PER)
│   └── dql_agent.py         # Double DQN ajanı
├── training/
│   ├── __init__.py
│   └── train.py             # Eğitim döngüsü (CLI + argparse)
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI WebSocket servisi
├── tools/
│   └── export_onnx.py       # .pth → .onnx dönüştürücü
├── maps/
│   ├── map_generator.py     # Rastgele harita üretici
│   └── map_001..100.json    # Önceden üretilmiş haritalar
├── models/                  # Eğitim sonucu .pth ve .onnx dosyaları
├── requirements.txt
└── setup.sh
```

---

## Kurulum

```bash
bash setup.sh
```

Veya adım adım:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Kullanım

### 1 — Harita Havuzu Oluştur

```bash
python maps/map_generator.py --num 100 --size 15 --ratio 0.15
```

### 2 — Modeli Eğit

```bash
# Temel eğitim (2000 bölüm)
python training/train.py

# Daha uzun eğitim, büyük grid
python training/train.py --episodes 5000 --grid-size 20 --obstacle-ratio 0.20

# Kaldığı yerden devam
python training/train.py --resume
```

**CLI parametreleri:**

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--episodes` | 2000 | Episode sayısı |
| `--grid-size` | 15 | Grid boyutu (NxN) |
| `--obstacle-ratio` | 0.15 | Engel yoğunluğu |
| `--lr` | 0.001 | Öğrenme hızı |
| `--gamma` | 0.95 | İndirim faktörü |
| `--batch-size` | 64 | Mini-batch boyutu |
| `--resume` | False | Var olan modelden devam et |
| `--no-random-maps` | False | Sabit harita kullan |

### 3 — ONNX'e Dönüştür (Spring Boot için)

```bash
python tools/export_onnx.py
# → models/model.onnx üretilir

# Spring Boot'a kopyala:
cp models/model.onnx ../otonomBack/PointSense/model.onnx
```

### 4 — FastAPI Sunucusu Başlat

```bash
python api/main.py
# → http://localhost:8001
# → ws://localhost:8001/ws/simulate
```

**Endpointler:**

| Method | Path | Açıklama |
|---|---|---|
| GET | `/health` | Servis durumu |
| POST | `/train` | Eğitim başlat |
| GET | `/train/status` | Eğitim ilerlemesi |
| POST | `/train/stop` | Eğitimi durdur |
| POST | `/infer` | Tek adım inference |
| GET | `/model/stats` | Model istatistikleri |
| POST | `/maps/load` | GameMapDTO yükle |
| WS | `/ws/simulate` | Canlı simülasyon |

---

## Teknik Detaylar

### Durum Vektörü (12 eleman — Spring Boot uyumlu)

```
[0]  agent_col / (size-1)    — ajan x normalize
[1]  agent_row / (size-1)    — ajan y normalize
[2]  goal_col  / (size-1)    — hedef x normalize
[3]  goal_row  / (size-1)    — hedef y normalize
[4]  sensor_left             — sol anlık engel (0/1)
[5]  sensor_right            — sağ anlık engel (0/1)
[6]  sensor_up               — üst anlık engel (0/1)
[7]  sensor_down             — alt anlık engel (0/1)
[8]  dist_left  / (size-1)   — sol engele normalize mesafe
[9]  dist_right / (size-1)   — sağ engele normalize mesafe
[10] dist_up    / (size-1)   — üst engele normalize mesafe
[11] dist_down  / (size-1)   — alt engele normalize mesafe
```

### Aksiyon Uzayı (Spring Boot uyumlu)

| ID | Yön | Delta |
|---|---|---|
| 0 | LEFT ← | (row, col−1) |
| 1 | RIGHT → | (row, col+1) |
| 2 | UP ↑ | (row−1, col) |
| 3 | DOWN ↓ | (row+1, col) |

### Ödül Mekanizması

| Durum | Ödül |
|---|---|
| Hedefe ulaşma | **+100** |
| Engele çarpma | **−50** |
| Sınır dışına çıkma | **−10** |
| Hedefe yaklaşma | **+1** |
| Hedeften uzaklaşma | **−0.5** |
| Her adım cezası | **−0.1** |
| Maksimum adım | **−1** |

### Q-Network Mimarisi (Dueling DQN)

```
Giriş (12)
   │
Dense(64, ReLU) → Dense(64, ReLU)
   │
   ├── Value stream:     Dense(32, ReLU) → V(s) [skaler]
   └── Advantage stream: Dense(32, ReLU) → A(s,a) [4 değer]
                               │
                    Q(s,a) = V(s) + A(s,a) − mean(A)
```

### Hiperparametreler

| Parametre | Değer | Açıklama |
|---|---|---|
| `learning_rate` | 0.001 | Adam optimizer |
| `gamma` | 0.95 | İndirim faktörü |
| `epsilon_start` | 1.0 | Başlangıç keşif oranı |
| `epsilon_min` | 0.01 | Minimum keşif |
| `epsilon_decay` | 0.995 | Episode başına azalma |
| `batch_size` | 64 | Mini-batch boyutu |
| `buffer_capacity` | 10,000 | Replay buffer kapasitesi |
| `target_update` | 10 | Hedef ağ güncelleme sıklığı |

---

## Geliştirme Yol Haritası

- [x] Faz 1 — Grid ortamı ve ödül fonksiyonu
- [x] Faz 2 — Dueling Double DQN ajanı
- [x] Faz 3 — Spring Boot ONNX entegrasyonu
- [x] Faz 4 — FastAPI WebSocket servisi
- [ ] Faz 5 — Hareketli engel desteği
- [ ] Faz 6 — Prioritized Experience Replay (PER) aktif kullanım
