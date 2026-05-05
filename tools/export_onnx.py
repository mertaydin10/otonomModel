"""
tools/export_onnx.py — PyTorch modelini ONNX formatına dönüştür

Spring Boot backend (ModelInferenceService.java), modeli ONNX runtime üzerinden
çalıştırır. Bu script eğitilmiş .pth dosyasını model.onnx'e aktarır.

Beklenen ONNX formatı (Spring Boot uyumlu):
  Giriş:  float32[1, 12]  — 12 elemanlı normalize durum vektörü
  Çıkış:  float32[1,  4]  — Q değerleri: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN

Kullanım:
  python tools/export_onnx.py
  python tools/export_onnx.py --input models/best_model.pth --output model.onnx
  python tools/export_onnx.py --validate   # ONNX Runtime ile doğrula
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from agent import DQLAgent


def export(
    pth_path: str,
    onnx_path: str,
    state_size: int = 12,
    action_size: int = 4,
    validate: bool = True,
) -> None:
    print(f"\n{'='*55}")
    print(f"  ONNX Export")
    print(f"{'='*55}")
    print(f"  Kaynak  : {pth_path}")
    print(f"  Hedef   : {onnx_path}")
    print(f"  State   : {state_size}")
    print(f"  Actions : {action_size}")

    # ── Model yükle ───────────────────────────────────────────────────────────
    agent = DQLAgent(state_size=state_size, action_size=action_size)
    if not agent.load(pth_path):
        print("\n[ERROR] Model dosyası bulunamadı. Önce eğitimi çalıştırın:")
        print("  python training/train.py --episodes 2000")
        sys.exit(1)

    net = agent.q_network
    net.eval()

    # ── ONNX dışa aktarma ────────────────────────────────────────────────────
    dummy = torch.zeros(1, state_size, dtype=torch.float32).to(agent.device)

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)

    torch.onnx.export(
        net,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["q_values"],
        dynamic_axes={
            "input":    {0: "batch_size"},
            "q_values": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"\n  ✅ ONNX dışa aktarıldı: {onnx_path}")

    # ── Doğrulama (opsiyonel) ─────────────────────────────────────────────────
    if validate:
        _validate(net, onnx_path, state_size, agent.device)


def _validate(net: torch.nn.Module, onnx_path: str, state_size: int, device) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnx veya onnxruntime yüklü değil, doğrulama atlandı.")
        print("         pip install onnx onnxruntime")
        return

    # ONNX model yapısını kontrol et
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("  ✅ ONNX model yapısı geçerli")

    # Runtime ile çıkarım karşılaştır
    test_input = np.random.rand(1, state_size).astype(np.float32)

    with torch.no_grad():
        pt_out = net(torch.tensor(test_input).to(device)).cpu().numpy()

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"input": test_input})[0]

    max_diff = float(np.abs(pt_out - ort_out).max())
    print(f"  PyTorch çıktısı : {np.round(pt_out[0], 4)}")
    print(f"  ONNX RT çıktısı : {np.round(ort_out[0], 4)}")
    print(f"  Maksimum fark   : {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✅ Doğrulama başarılı — Spring Boot'a kopyalanabilir")
    else:
        print("  ⚠️  Fark büyük, kontrol edin")

    print(f"\n  Spring Boot'a kopyalama:")
    spring_model = os.path.join(
        os.path.dirname(os.path.dirname(onnx_path)),
        "otonomBack", "PointSense", "model.onnx"
    )
    print(f"    cp {onnx_path} {spring_model}")
    print(f"{'='*55}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch → ONNX dönüştürücü")
    parser.add_argument(
        "--input", default="models/best_model.pth",
        help="Kaynak .pth dosyası (varsayılan: models/best_model.pth)"
    )
    parser.add_argument(
        "--output", default="models/model.onnx",
        help="Hedef .onnx dosyası (varsayılan: models/model.onnx)"
    )
    parser.add_argument(
        "--state-size", type=int, default=12,
        help="Durum vektörü boyutu (varsayılan: 12)"
    )
    parser.add_argument(
        "--validate", action="store_true", default=True,
        help="ONNX Runtime ile doğrulama yap (varsayılan: True)"
    )
    parser.add_argument(
        "--no-validate", dest="validate", action="store_false",
        help="Doğrulamayı atla"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    export(
        pth_path=args.input,
        onnx_path=args.output,
        state_size=args.state_size,
        validate=args.validate,
    )
