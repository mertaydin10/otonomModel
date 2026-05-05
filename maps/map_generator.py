import os
import sys
import json
import argparse
from tqdm import tqdm
import numpy as np
import hashlib

# Betiğin bir üst dizinini (proje kök dizini) Python'un arama yoluna ekle.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from environment.grid_env import GridEnvironment

def to_cartesian(pos, grid_size):
    center = grid_size // 2
    col, row = pos
    x = col - center
    y = -(row - center)
    return int(x), int(y)

def get_map_hash(map_data):
    map_string = json.dumps(map_data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(map_string.encode('utf-8')).hexdigest()

def load_existing_hashes(output_dir):
    hashes = set()
    if not os.path.exists(output_dir):
        return hashes, 0
    
    max_num = 0
    for filename in os.listdir(output_dir):
        if filename.startswith("map_") and filename.endswith(".json"):
            try:
                num = int(filename.split('_')[1].split('.')[0])
                if num > max_num:
                    max_num = num
                
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    hashes.add(get_map_hash(data))
            except (IndexError, ValueError, json.JSONDecodeError):
                continue
    return hashes, max_num

def migrate_maps_to_new_format(output_dir: str):
    final_output_dir = os.path.join(project_root, output_dir)
    if not os.path.exists(final_output_dir):
        print(f"'{final_output_dir}' klasörü bulunamadı. Dönüştürülecek harita yok.")
        return

    print(f"'{final_output_dir}' klasöründeki haritalar yeni formata dönüştürülüyor...")
    converted_count = 0
    for filename in tqdm(os.listdir(final_output_dir), desc="Haritalar Dönüştürülüyor"):
        if not (filename.startswith("map_") and filename.endswith(".json")):
            continue

        file_path = os.path.join(final_output_dir, filename)
        with open(file_path, 'r+') as f:
            try:
                data = json.load(f)
                if "map_name" not in data:
                    new_data = {
                        "map_name": data.get("mapName", filename.replace('.json', '')),
                        "grid_size": {"x": data["gridSize"]["width"], "y": data["gridSize"]["height"]},
                        "start_pos": data["startPos"],
                        "target_pos": data["targetPos"],
                        "obstacles": {
                            "static": [
                                {"x": obs["x"], "y": obs["y"], "w": 1, "h": 1} for obs in data["obstacles"]
                            ],
                            "dynamic": []
                        }
                    }
                    f.seek(0)
                    json.dump(new_data, f, indent=2)
                    f.truncate()
                    converted_count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"\n'{filename}' dosyası işlenirken hata oluştu, atlanıyor: {e}")
                continue
    
    print(f"\nİşlem tamamlandı. {converted_count} harita başarıyla yeni formata dönüştürüldü.")


def generate_map_pool(num_maps: int, grid_size: int, obstacle_ratio: float, output_dir: str):
    if grid_size % 2 == 0:
        print(f"Uyarı: grid_size ({grid_size}) çift sayı. Kartezyen dönüşüm için tek sayı olması önerilir.")

    final_output_dir = os.path.join(project_root, output_dir)
    existing_hashes, start_index = load_existing_hashes(final_output_dir)
    print(f"Mevcut harita havuzunda {len(existing_hashes)} benzersiz harita bulundu.")
    if start_index > 0:
        print(f"Yeni haritalar 'map_{start_index + 1:03d}.json' dosyasından başlayarak oluşturulacak.")

    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    env = GridEnvironment(size=grid_size, obstacle_ratio=obstacle_ratio, random_maps=True)
    generated_count = 0
    pbar = tqdm(total=num_maps, desc="Benzersiz Harita Üretiliyor")
    while generated_count < num_maps:
        env.reset()
        obstacle_indices = np.argwhere(env.grid == 1)
        start_cartesian = to_cartesian(env.start_pos, env.size)
        goal_cartesian = to_cartesian(env.goal_pos, env.size)
        obstacles_cartesian = [to_cartesian(pos, env.size) for pos in obstacle_indices]
        current_map_index = start_index + generated_count + 1
        
        map_data = {
            "map_name": f"generated_map_{current_map_index:03d}",
            "grid_size": {"x": env.size, "y": env.size},
            "start_pos": {"x": start_cartesian[0], "y": start_cartesian[1]},
            "target_pos": {"x": goal_cartesian[0], "y": goal_cartesian[1]},
            "obstacles": {
                "static": [{"x": pos[0], "y": pos[1], "w": 1, "h": 1} for pos in obstacles_cartesian],
                "dynamic": []
            }
        }
        
        map_hash = get_map_hash(map_data)
        if map_hash in existing_hashes:
            continue

        existing_hashes.add(map_hash)
        file_name = f"map_{current_map_index:03d}.json"
        file_path = os.path.join(final_output_dir, file_name)
        with open(file_path, 'w') as f:
            json.dump(map_data, f, indent=2)
        generated_count += 1
        pbar.update(1)
    
    pbar.close()
    print(f"\nBaşarıyla {generated_count} yeni ve benzersiz harita üretildi.")
    print(f"Haritalar '{final_output_dir}' klasörüne kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Otonom Sürüş için Harita Havuzu Üretici/Dönüştürücü")
    parser.add_argument("--num", type=int, default=50, help="Üretilecek harita sayısı.")
    parser.add_argument("--size", type=int, default=15, help="Haritaların boyutu (NxN). Tek sayı olmalı.")
    parser.add_argument("--ratio", type=float, default=0.2, help="Engel yoğunluğu (0.0 ile 1.0 arası).")
    parser.add_argument("--out", type=str, default="maps", help="Çıktı klasörünün adı (proje kök dizinine göre).")
    parser.add_argument("--migrate", action="store_true", help="Mevcut haritaları yeni formata dönüştürür.")
    
    args = parser.parse_args()

    if args.migrate:
        migrate_maps_to_new_format(args.out)
    else:
        generate_map_pool(
            num_maps=args.num,
            grid_size=args.size,
            obstacle_ratio=args.ratio,
            output_dir=args.out
        )