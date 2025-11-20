#!/usr/bin/env python3
"""
修正版數據準備 - 按 lot 分割避免數據洩漏
"""

from pathlib import Path
import random
import shutil
from collections import defaultdict

def collect_samples_by_lot(data_root):
    """
    按 lot 收集樣本（而非按影像）
    """
    data_root = Path(data_root)
    
    lots_data = defaultdict(list)  # {lot_id: [samples]}
    
    print("="*80)
    print("按 Lot 收集數據")
    print("="*80)
    
    for crystal in ['100', '111', '776']:
        crystal_path = data_root / crystal
        
        if not crystal_path.exists():
            continue
        
        print(f"\n處理 {crystal} 晶向...")
        
        lots = [d for d in crystal_path.iterdir() if d.is_dir()]
        
        for lot in lots:
            lot_id = f"{crystal}_{lot.name}"
            
            images_dir = lot / "images"
            labels_dir = lot / "labels"
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            # 收集這個 lot 的所有有效樣本
            lot_samples = []
            
            for label_file in labels_dir.glob("*.txt"):
                if label_file.name == "classes.txt":
                    continue
                
                img_file = images_dir / f"{label_file.stem}.jpg"
                
                if not img_file.exists():
                    continue
                
                # 驗證標註格式
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    valid = True
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:  # class x y w h
                            valid = False
                            break
                    
                    if valid and len(lines) > 0:
                        lot_samples.append({
                            'image': img_file,
                            'label': label_file,
                            'crystal': crystal,
                            'lot': lot.name
                        })
                
                except Exception as e:
                    print(f"  ⚠ 跳過錯誤標註: {label_file.name}")
                    continue
            
            if len(lot_samples) > 0:
                lots_data[lot_id] = lot_samples
                print(f"  {lot.name}: {len(lot_samples)} 張影像")
    
    return lots_data


def split_by_lot(lots_data, train_ratio=0.8, seed=42):
    """
    按 lot 分割（避免數據洩漏）
    """
    random.seed(seed)
    
    # 獲取所有 lot IDs
    lot_ids = list(lots_data.keys())
    random.shuffle(lot_ids)
    
    # 計算分割點
    split_idx = int(len(lot_ids) * train_ratio)
    
    train_lot_ids = lot_ids[:split_idx]
    val_lot_ids = lot_ids[split_idx:]
    
    # 收集樣本
    train_samples = []
    val_samples = []
    
    for lot_id in train_lot_ids:
        train_samples.extend(lots_data[lot_id])
    
    for lot_id in val_lot_ids:
        val_samples.extend(lots_data[lot_id])
    
    print("\n" + "="*80)
    print("分割統計")
    print("="*80)
    print(f"總 lots 數: {len(lot_ids)}")
    print(f"訓練 lots: {len(train_lot_ids)} ({len(train_lot_ids)/len(lot_ids)*100:.1f}%)")
    print(f"驗證 lots: {len(val_lot_ids)} ({len(val_lot_ids)/len(lot_ids)*100:.1f}%)")
    print(f"\n訓練樣本數: {len(train_samples)}")
    print(f"驗證樣本數: {len(val_samples)}")
    
    # 各晶向統計
    print(f"\n各晶向分布:")
    for split_name, samples in [("訓練集", train_samples), ("驗證集", val_samples)]:
        crystal_counts = defaultdict(int)
        for s in samples:
            crystal_counts[s['crystal']] += 1
        
        print(f"  {split_name}:")
        for crystal in ['100', '111', '776']:
            count = crystal_counts[crystal]
            pct = count / len(samples) * 100 if len(samples) > 0 else 0
            print(f"    {crystal}: {count} ({pct:.1f}%)")
    
    return train_samples, val_samples, train_lot_ids, val_lot_ids


def create_dataset_structure(train_samples, val_samples, output_dir):
    """
    建立 YOLO 格式的數據集結構
    """
    output_dir = Path(output_dir)
    
    # 建立目錄結構
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("建立數據集")
    print("="*80)
    
    # 處理訓練集
    print("\n處理訓練集...")
    for idx, sample in enumerate(train_samples):
        src_img = sample['image']
        src_label = sample['label']
        
        # 使用唯一命名：crystal_lot_原檔名
        base_name = f"{sample['crystal']}_{sample['lot']}_{src_img.stem}"
        
        dst_img = output_dir / 'train' / 'images' / f"{base_name}.jpg"
        dst_label = output_dir / 'train' / 'labels' / f"{base_name}.txt"
        
        # 建立符號連結或複製
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())
        if not dst_label.exists():
            dst_label.symlink_to(src_label.resolve())
    
    # 處理驗證集
    print("處理驗證集...")
    for idx, sample in enumerate(val_samples):
        src_img = sample['image']
        src_label = sample['label']
        
        base_name = f"{sample['crystal']}_{sample['lot']}_{src_img.stem}"
        
        dst_img = output_dir / 'val' / 'images' / f"{base_name}.jpg"
        dst_label = output_dir / 'val' / 'labels' / f"{base_name}.txt"
        
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())
        if not dst_label.exists():
            dst_label.symlink_to(src_label.resolve())
    
    print(f"\n✓ 數據集已建立於: {output_dir}")
    
    # 建立 data.yaml
    yaml_content = f"""# Project A Dataset Configuration
# 按 lot 分割避免數據洩漏

path: {output_dir.resolve()}
train: train/images
val: val/images

nc: 1
names: ['breakline']

# 數據集統計
# 訓練集: {len(train_samples)} 張影像
# 驗證集: {len(val_samples)} 張影像
"""
    
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ 配置檔已建立: {yaml_path}")


def analyze_leakage_risk(train_lot_ids, val_lot_ids):
    """
    分析是否有數據洩漏風險
    """
    print("\n" + "="*80)
    print("數據洩漏風險分析")
    print("="*80)
    
    # 檢查是否有重疊
    train_set = set(train_lot_ids)
    val_set = set(val_lot_ids)
    overlap = train_set & val_set
    
    if len(overlap) > 0:
        print(f"❌ 發現重疊！{len(overlap)} 個 lots 同時出現在訓練和驗證集")
        for lot_id in list(overlap)[:5]:
            print(f"  - {lot_id}")
    else:
        print(f"✓ 無重疊！訓練集和驗證集完全分離")
    
    print(f"\n分離程度:")
    print(f"  訓練 lots: {len(train_set)}")
    print(f"  驗證 lots: {len(val_set)}")
    print(f"  總數: {len(train_set) + len(val_set)}")


if __name__ == "__main__":
    print("="*80)
    print("修正版數據準備 - 按 Lot 分割")
    print("="*80)
    
    # 配置
    data_root = Path("/TOPIC/projectA/A_training_1")
    output_dir = Path("/home/114078/projectA_work/dataset_correct")
    
    # 步驟 1: 按 lot 收集數據
    lots_data = collect_samples_by_lot(data_root)
    
    if len(lots_data) == 0:
        print("\n❌ 沒有找到有效數據！")
        exit(1)
    
    # 步驟 2: 按 lot 分割
    train_samples, val_samples, train_lot_ids, val_lot_ids = split_by_lot(
        lots_data, 
        train_ratio=0.8,
        seed=42
    )
    
    # 步驟 3: 分析洩漏風險
    analyze_leakage_risk(train_lot_ids, val_lot_ids)
    
    # 步驟 4: 建立數據集
    create_dataset_structure(train_samples, val_samples, output_dir)
    
    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"\n數據集位置: {output_dir}")
    print(f"配置檔: {output_dir / 'data.yaml'}")