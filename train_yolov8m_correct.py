#!/usr/bin/env python3
"""
訓練 YOLOv8m - 修正版數據集
保守參數避免 OOM + 早停策略
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path
from datetime import datetime

# 設置離線模式
os.environ['YOLO_OFFLINE'] = '1'

def train_model():
    """
    使用修正版數據集訓練 YOLOv8m
    """
    
    print("="*80)
    print("YOLOv8m 訓練 - 修正版（無數據洩漏）")
    print("="*80)
    
    # 檢查 GPU
    print(f"\nCUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 檢查數據集
    data_yaml = Path("/home/114078/projectA_work/dataset_correct/data.yaml")
    
    if not data_yaml.exists():
        print(f"\n❌ 錯誤：找不到數據集配置檔")
        print(f"請先執行: python3 prepare_dataset_correct.py")
        return
    
    print(f"\n✓ 數據集配置: {data_yaml}")
    
    # 載入預訓練模型
    print(f"\n載入 YOLOv8-medium 預訓練模型...")
    
    # 檢查是否有本地權重
    local_weights = Path("/home/114078/projectA_work/yolov8m.pt")
    
    if local_weights.exists():
        print(f"✓ 使用本地權重: {local_weights}")
        model = YOLO(str(local_weights))
    else:
        print(f"⚠ 本地權重不存在，使用預設模型")
        model = YOLO('yolov8m.pt')
    
    # 訓練配置 - 保守設定避免 OOM
    print(f"\n開始訓練...")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n配置:")
    print(f"  batch_size: 12 (保守設定避免 OOM)")
    print(f"  patience: 15 (早停策略)")
    print(f"  預估時間: 8-10 小時")
    
    results = model.train(
        # 數據配置
        data=str(data_yaml),
        epochs=100,
        imgsz=1024,
        batch=12,  # 保守設定：12 而非 16
        workers=0,  # 容器環境限制
        
        # 小目標優化
        hsv_h=0.010,
        hsv_s=0.5,
        hsv_v=0.3,
        
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        
        # 損失函數權重
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 訓練策略
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # 其他設置
        close_mosaic=10,
        amp=True,
        patience=15,  # 早停：15 epochs 無改進就停
        save=True,
        save_period=-1,
        
        # 輸出設置
        project='/home/114078/projectA_work/results',
        name='yolov8m_correct',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=False,
        single_cls=True,
        rect=False,
        cos_lr=True,
        
        device=0
    )
    
    print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存最佳模型到固定位置
    best_model = Path("/home/114078/projectA_work/results/yolov8m_correct/weights/best.pt")
    
    if best_model.exists():
        target = Path("/home/114078/projectA_work/models/best_model_yolov8m_correct.pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy(best_model, target)
        
        print(f"\n✓ 最佳模型已複製至: {target}")
    
    print("\n" + "="*80)
    print("訓練完成！")
    print("="*80)
    
    # 執行驗證
    print("\n執行最終驗證...")
    
    try:
        metrics = model.val(
            data=str(data_yaml),
            imgsz=1024,
            batch=12,
            workers=0,
            device=0
        )
        
        print("\n最終指標:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        
    except Exception as e:
        print(f"\n驗證時發生錯誤: {e}")
        print("但模型已訓練完成，可以手動驗證")
    
    return results


if __name__ == "__main__":
    try:
        print("\n" + "="*80)
        print("⚠️ 重要提醒")
        print("="*80)
        print("YOLOv8m 訓練預估需要 8-10 小時")
        print("建議使用 nohup 背景執行:")
        print("  nohup python3 train_yolov8m_correct.py > train_yolov8m_correct.log 2>&1 &")
        print("\n按 Ctrl+C 取消，或等待 5 秒後自動開始...")
        print("="*80 + "\n")
        
        import time
        time.sleep(5)
        
        results = train_model()
        
        print("\n" + "="*80)
        print("下一步:")
        print("="*80)
        print("1. 評估模型: python3 evaluate_competition_metric_correct.py")
        print("   (記得改 model_path 為 best_model_yolov8m_correct.pt)")
        print("2. 比較 nano vs medium 的差異")
        
    except KeyboardInterrupt:
        print("\n\n訓練被中斷")
    except Exception as e:
        print(f"\n❌ 訓練過程發生錯誤:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
