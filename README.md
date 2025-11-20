# Project A - 訓練流程說明

## 目錄結構
```
/home/114078/projectA_work/
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
cd /home/114078/projectA_work
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
tail -f /home/114078/projectA_work/results/yolov8n_breakline/train/results.txt
```

### 3. 驗證模型
訓練完成後，查看驗證結果：
```bash
cat /home/114078/projectA_work/results/yolov8n_breakline/val/results.txt
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

生成時間: 2025-11-12 17:29:54
