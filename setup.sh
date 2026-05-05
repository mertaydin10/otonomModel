#!/bin/bash
# setup.sh — Python AI Servis kurulum ve başlatma scripti

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════════"
echo "  🚗 Otonom Sürüş AI Servis Kurulumu"
echo "═══════════════════════════════════════════════════"

# ── Virtual environment ──────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
  echo "▶ Virtual environment oluşturuluyor..."
  python3 -m venv venv
fi

source venv/bin/activate
echo "✓ Virtual environment aktif"

# ── Bağımlılıklar ─────────────────────────────────────────────────────────────
echo "▶ Bağımlılıklar yükleniyor..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Bağımlılıklar yüklendi"

# ── Klasörler ─────────────────────────────────────────────────────────────────
mkdir -p models maps
echo "✓ models/ ve maps/ klasörleri hazır"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Kullanım Kılavuzu"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  1. Harita havuzu oluştur (100 harita):"
echo "     python maps/map_generator.py --num 100 --size 15"
echo ""
echo "  2. Modeli eğit:"
echo "     python training/train.py --episodes 2000"
echo "     python training/train.py --episodes 5000 --grid-size 20"
echo ""
echo "  3. Eğitimi devam ettir:"
echo "     python training/train.py --resume"
echo ""
echo "  4. ONNX'e dönüştür (Spring Boot için):"
echo "     python tools/export_onnx.py"
echo "     → models/model.onnx üretilir"
echo "     → otonomBack/PointSense/ klasörüne kopyala"
echo ""
echo "  5. FastAPI sunucusunu başlat:"
echo "     python api/main.py"
echo "     → http://localhost:8001"
echo "     → WS: ws://localhost:8001/ws/simulate"
echo "═══════════════════════════════════════════════════"
