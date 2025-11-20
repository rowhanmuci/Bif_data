#!/usr/bin/env python3
"""
訓練腳本 v3 - 使用修正版數據集
避免數據洩漏 + 過濾錯誤標註
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
    使用修正版數據集訓練模型
    """
    
    print("="*80)
    print("YOLOv8 訓練 - 修正版（無數據洩漏）")
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
    print(f"\n載入 YOLOv8-nano 預訓練模型...")
    
    # 檢查是否有本地權重
    local_weights = Path("/home/114078/projectA_work/yolov8n.pt")
    
    if local_weights.exists():
        print(f"✓ 使用本地權重: {local_weights}")
        model = YOLO(str(local_weights))
    else:
        print(f"⚠ 本地權重不存在，嘗試載入預設模型")
        model = YOLO('yolov8n.pt')
    
    # 訓練配置
    print(f"\n開始訓練...")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = model.train(
        # 數據配置
        data=str(data_yaml),
        epochs=100,
        imgsz=1024,
        batch=16,
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
        patience=20,
        save=True,
        save_period=-1,
        
        # 輸出設置
        project='/home/114078/projectA_work/results',
        name='yolov8n_correct',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        seed=42,  # 固定隨機種子
        deterministic=False,
        single_cls=True,
        rect=False,
        cos_lr=True,
        
        device=0
    )
    
    print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存最佳模型到固定位置
    best_model = Path("/home/114078/projectA_work/results/yolov8n_correct/weights/best.pt")
    
    if best_model.exists():
        target = Path("/home/114078/projectA_work/models/best_model_correct.pt")
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
            batch=16,
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
        results = train_model()
        
        print("\n" + "="*80)
        print("下一步:")
        print("="*80)
        print("1. 評估模型: python3 evaluate_competition_metric.py")
        print("2. 測試推論: python3 inference_to_csv.py test")
        print("3. 比較新舊模型的差異")
        
    except KeyboardInterrupt:
        print("\n\n訓練被中斷")
    except Exception as e:
        print(f"\n❌ 訓練過程發生錯誤:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()