#!/usr/bin/env python3
"""
Project A - 完整訓練流程
矽單晶斷線檢測 - 基於 YOLOv8 的解決方案
"""

import os
from pathlib import Path
import yaml
import shutil
from datetime import datetime

class ProjectAPipeline:
    def __init__(self, work_dir="/home/114078/projectA_work"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        self.data_root = Path("/TOPIC/projectA/A_training_1")
        self.crystal_types = ["100", "111", "776"]
        
        # 建立工作目錄結構
        self.dataset_dir = self.work_dir / "dataset"
        self.models_dir = self.work_dir / "models"
        self.results_dir = self.work_dir / "results"
        
        for dir_path in [self.dataset_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def prepare_dataset(self):
        """準備訓練數據集 - YOLO 格式"""
        
        print("\n" + "="*80)
        print("Step 1: 準備數據集")
        print("="*80)
        
        # 建立 YOLO 格式的目錄結構
        for split in ['train', 'val']:
            for subdir in ['images', 'labels']:
                (self.dataset_dir / split / subdir).mkdir(parents=True, exist_ok=True)
        
        # 複製並分割數據
        import random
        random.seed(42)
        
        all_samples = []
        
        for crystal in self.crystal_types:
            print(f"\n處理 {crystal} 晶向...")
            crystal_path = self.data_root / crystal
            
            lots = [d for d in crystal_path.iterdir() if d.is_dir()]
            
            for lot in lots:
                images_dir = lot / "images"
                labels_dir = lot / "labels"
                
                if not images_dir.exists() or not labels_dir.exists():
                    continue
                
                # 只使用有標註的影像（訓練 positive samples）
                label_files = [f for f in labels_dir.glob("*.txt") 
                             if f.name != "classes.txt"]
                
                for label_file in label_files:
                    img_file = images_dir / f"{label_file.stem}.jpg"
                    
                    if img_file.exists():
                        all_samples.append({
                            'image': img_file,
                            'label': label_file,
                            'crystal': crystal
                        })
        
        print(f"\n總共收集到 {len(all_samples)} 個有斷線的樣本")
        
        # 80-20 分割
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        print(f"訓練集: {len(train_samples)} 樣本")
        print(f"驗證集: {len(val_samples)} 樣本")
        
        # 複製檔案（建立符號連結以節省空間）
        for split_name, samples in [('train', train_samples), ('val', val_samples)]:
            print(f"\n建立 {split_name} 集的符號連結...")
            
            for idx, sample in enumerate(samples):
                # 建立唯一檔名：crystal_原檔名
                img_name = f"{sample['crystal']}_{sample['image'].name}"
                label_name = f"{sample['crystal']}_{sample['label'].name}"
                
                img_dst = self.dataset_dir / split_name / 'images' / img_name
                label_dst = self.dataset_dir / split_name / 'labels' / label_name
                
                # 建立符號連結
                if not img_dst.exists():
                    os.symlink(sample['image'], img_dst)
                if not label_dst.exists():
                    os.symlink(sample['label'], label_dst)
                
                if (idx + 1) % 500 == 0:
                    print(f"  已處理 {idx + 1}/{len(samples)} 個樣本")
        
        # 建立 data.yaml
        data_yaml = {
            'path': str(self.dataset_dir),
            'train': 'train/images',
            'val': 'val/images',
            'names': {0: 'breakline'},
            'nc': 1
        }
        
        yaml_path = self.dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"\n✓ 數據集準備完成")
        print(f"  配置檔: {yaml_path}")
        
        return yaml_path
    
    def create_training_config(self):
        """建立訓練配置"""
        
        print("\n" + "="*80)
        print("Step 2: 建立訓練配置")
        print("="*80)
        
        # 針對小目標檢測的優化配置
        config = """
# YOLOv8 訓練配置 - 針對小目標優化

# 基本設定
model: yolov8n.pt  # 使用 nano 模型作為起點
data: {data_yaml}

# 訓練參數
epochs: 100
batch: 16  # 根據 GPU 記憶體調整
imgsz: 1024  # 高解析度以保留小目標細節
patience: 20  # Early stopping

# 優化器
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# 數據增強（適度，避免破壞小目標）
hsv_h: 0.010
hsv_s: 0.5
hsv_v: 0.3
degrees: 5.0  # 小角度旋轉
translate: 0.1
scale: 0.3
shear: 2.0
flipud: 0.0  # 不上下翻轉（晶線有方向性）
fliplr: 0.5
mosaic: 0.5  # 降低 mosaic 比例（保護小目標）

# 小目標優化
amp: True  # Mixed precision training
close_mosaic: 10  # 最後 10 epochs 關閉 mosaic

# 損失函數權重
box: 7.5  # 提高 box loss 權重（小目標）
cls: 0.5
dfl: 1.5

# 其他
workers: 4
device: 0
exist_ok: True
pretrained: True
verbose: True
save: True
save_period: 10
val: True
plots: True
"""
        
        config_path = self.work_dir / "train_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config.format(data_yaml=self.dataset_dir / 'data.yaml'))
        
        print(f"✓ 訓練配置已建立: {config_path}")
        
        return config_path
    
    def create_training_script(self, data_yaml):
        """建立訓練腳本"""
        
        print("\n" + "="*80)
        print("Step 3: 建立訓練腳本")
        print("="*80)
        
        script = f"""#!/usr/bin/env python3
'''
YOLOv8 訓練腳本 - Project A
'''

from ultralytics import YOLO
import torch

# 檢查 GPU
print(f"CUDA 可用: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"GPU: {{torch.cuda.get_device_name(0)}}")

# 載入模型
model = YOLO('yolov8n.pt')

# 訓練
results = model.train(
    data='{data_yaml}',
    epochs=100,
    imgsz=1024,
    batch=16,
    
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
    lr0=0.001,
    patience=20,
    project='{self.results_dir}',
    name='yolov8n_breakline',
    exist_ok=True,
    verbose=True,
    plots=True,
    device=0
)

# 驗證
metrics = model.val()
print(f"\\nmAP50: {{metrics.box.map50:.4f}}")
print(f"mAP50-95: {{metrics.box.map:.4f}}")

# 保存最終模型
model.save('{self.models_dir}/best_model.pt')
print(f"\\n模型已保存至: {self.models_dir}/best_model.pt")
"""
        
        script_path = self.work_dir / "train.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        
        print(f"✓ 訓練腳本已建立: {script_path}")
        
        return script_path
    
    def create_inference_script(self):
        """建立推論腳本"""
        
        script = f"""#!/usr/bin/env python3
'''
推論腳本 - 生成測試集預測結果
'''

from ultralytics import YOLO
from pathlib import Path
import shutil

# 載入訓練好的模型
model = YOLO('{self.models_dir}/best_model.pt')

# 測試數據路徑（決賽時會提供）
test_root = Path("/TOPIC/projectA/A_test")  # 請根據實際路徑調整

# 輸出目錄
output_dir = Path('{self.work_dir}/predictions')
output_dir.mkdir(exist_ok=True)

# 對每個晶向進行預測
for crystal in ['100', '111', '776']:
    crystal_path = test_root / crystal
    
    if not crystal_path.exists():
        print(f"警告: {{crystal}} 路徑不存在")
        continue
    
    # 建立輸出目錄結構（與訓練數據相同）
    lots = [d for d in crystal_path.iterdir() if d.is_dir()]
    
    for lot in lots:
        images_dir = lot / "images"
        
        if not images_dir.exists():
            continue
        
        # 建立對應的 labels 輸出目錄
        output_labels_dir = output_dir / crystal / lot.name / "labels"
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 對所有影像進行預測
        image_files = sorted(images_dir.glob("*.jpg"))
        
        for img_file in image_files:
            results = model.predict(
                source=str(img_file),
                conf=0.25,  # 置信度閾值
                iou=0.45,
                device=0,
                verbose=False
            )
            
            # 保存預測結果（YOLO 格式）
            result = results[0]
            
            if len(result.boxes) > 0:
                # 有檢測到斷線
                label_file = output_labels_dir / f"{{img_file.stem}}.txt"
                
                with open(label_file, 'w') as f:
                    for box in result.boxes:
                        # 轉換為 YOLO 格式
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        conf = box.conf[0].item()
                        
                        # class x_center y_center width height
                        f.write(f"0 {{x_center:.6f}} {{y_center:.6f}} {{width:.6f}} {{height:.6f}}\\n")

print(f"\\n預測完成！結果保存在: {{output_dir}}")
"""
        
        script_path = self.work_dir / "inference.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        script_path.chmod(0o755)
        
        print(f"✓ 推論腳本已建立: {script_path}")
        
        return script_path
    
    def run(self):
        """執行完整流程"""
        
        print("\n" + "="*80)
        print("Project A - 完整訓練流程初始化")
        print("="*80)
        print(f"工作目錄: {self.work_dir}")
        print(f"數據來源: {self.data_root}")
        
        # 1. 準備數據集
        data_yaml = self.prepare_dataset()
        
        # 2. 建立訓練配置
        config_path = self.create_training_config()
        
        # 3. 建立訓練腳本
        train_script = self.create_training_script(data_yaml)
        
        # 4. 建立推論腳本
        inference_script = self.create_inference_script()
        
        # 5. 建立 README
        readme = f"""
# Project A - 訓練流程說明

## 目錄結構
```
{self.work_dir}/
├── dataset/              # YOLO 格式數據集
│   ├── train/
│   ├── val/
│   └── data.yaml
├── models/               # 訓練好的模型
├── results/              # 訓練結果和日誌
├── train.py              # 訓練腳本
└── inference.py          # 推論腳本
```

## 執行步驟

### 1. 訓練模型
```bash
cd {self.work_dir}
python3 train.py
```

訓練過程會：
- 使用 YOLOv8n 作為基礎
- 針對小目標進行優化
- 訓練 100 epochs（有 early stopping）
- 保存最佳模型

### 2. 監控訓練
查看訓練日誌：
```bash
tail -f {self.results_dir}/yolov8n_breakline/train/results.txt
```

### 3. 驗證模型
訓練完成後，查看驗證結果：
```bash
cat {self.results_dir}/yolov8n_breakline/val/results.txt
```

### 4. 測試集預測（決賽日）
```bash
python3 inference.py
```

## 重要提醒

1. **GPU 記憶體**：如果 batch=16 太大，降低至 8 或 4
2. **訓練時間**：預估 8-12 小時（100 epochs）
3. **Early Stopping**：如果 20 epochs 沒改善會自動停止
4. **最佳模型**：保存在 models/best_model.pt

## 下一步優化

如果 baseline 結果不理想：
1. 調整置信度閾值（conf）
2. 增加訓練數據（使用數據增強）
3. 嘗試更大的模型（yolov8s, yolov8m）
4. 實作時序平滑策略

生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        readme_path = self.work_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        print("\n" + "="*80)
        print("✓ 完整流程初始化完成！")
        print("="*80)
        
        print(f"\n下一步：開始訓練")
        print(f"  cd {self.work_dir}")
        print(f"  python3 train.py")
        
        print(f"\n詳細說明請查看: {readme_path}")


if __name__ == "__main__":
    pipeline = ProjectAPipeline()
    pipeline.run()