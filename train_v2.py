#!/usr/bin/env python3
"""
YOLOv8 訓練腳本 - Project A (無需下載版本)
使用本地配置從頭訓練，或使用上傳的預訓練權重
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import os

# 檢查 GPU
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 工作目錄
work_dir = Path("/home/114078/projectA_work")
data_yaml = work_dir / "dataset" / "data.yaml"

# 檢查數據集是否準備好
if not data_yaml.exists():
    print("錯誤：數據集尚未準備！")
    print("請先執行: python3 projectA_complete_pipeline.py")
    exit(1)

print(f"\n使用數據集: {data_yaml}")

# 方案選擇
weights_path = work_dir / "yolov8n.pt"

if weights_path.exists():
    print(f"\n✓ 找到預訓練權重: {weights_path}")
    print("使用遷移學習模式（推薦）")
    model = YOLO(str(weights_path))
    transfer_learning = True
else:
    print(f"\n⚠ 未找到預訓練權重: {weights_path}")
    print("將從頭訓練（scratch training）")
    print("\n建議：")
    print("1. 在本地電腦下載 yolov8n.pt")
    print("   下載連結: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt")
    print("2. 使用 FileZilla 上傳到平台")
    print(f"   目標路徑: {weights_path}")
    print("3. 重新執行此腳本\n")
    
    response = input("是否繼續從頭訓練？(y/n): ")
    if response.lower() != 'y':
        print("訓練取消")
        exit(0)
    
    # 從頭訓練 - 使用配置檔案
    model = YOLO('yolov8n.yaml')
    transfer_learning = False

print("\n" + "="*80)
print("開始訓練")
print("="*80)

# 訓練參數（針對 scratch 訓練調整）
if transfer_learning:
    # 遷移學習參數
    epochs = 100
    lr0 = 0.001
    patience = 20
else:
    # 從頭訓練參數（需要更多訓練）
    epochs = 200  # 增加 epochs
    lr0 = 0.01    # 提高學習率
    patience = 30  # 增加耐心
    print(f"\n⚠ 從頭訓練需要更長時間（約 {epochs} epochs）")

# 訓練
try:
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=1024,
        batch=16,  # 如果 OOM，改成 8 或 4
        
        # 小目標優化
        hsv_h=0.010,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        
        # 損失權重
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # 其他設定
        optimizer='AdamW',
        lr0=lr0,
        patience=patience,
        project=str(work_dir / "results"),
        name='yolov8n_breakline',
        exist_ok=True,
        verbose=True,
        plots=True,
        device=0,
        
        # 保存設定
        save=True,
        save_period=10,  # 每 10 epochs 保存一次
    )
    
    print("\n" + "="*80)
    print("訓練完成！")
    print("="*80)
    
    # 驗證
    print("\n執行最終驗證...")
    metrics = model.val()
    
    print(f"\n最終指標:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.p:.4f}")
    print(f"  Recall: {metrics.box.r:.4f}")
    
    # 保存最終模型
    model_save_path = work_dir / "models" / "best_model.pt"
    model_save_path.parent.mkdir(exist_ok=True)
    
    # 複製最佳模型
    import shutil
    best_model = work_dir / "results" / "yolov8n_breakline" / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy(best_model, model_save_path)
        print(f"\n✓ 最佳模型已保存至: {model_save_path}")
    
    print(f"\n訓練結果目錄: {work_dir / 'results' / 'yolov8n_breakline'}")
    
except KeyboardInterrupt:
    print("\n\n訓練被中斷")
    print("部分訓練的模型已保存")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n\n❌ GPU 記憶體不足！")
        print("解決方案：")
        print("1. 降低 batch size（改成 8 或 4）")
        print("2. 降低 imgsz（改成 640）")
        print("\n請修改 train_v2.py 的參數後重試")
    else:
        raise e

except Exception as e:
    print(f"\n\n❌ 訓練過程發生錯誤:")
    print(f"{type(e).__name__}: {e}")
    raise e