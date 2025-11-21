
#!/usr/bin/env python3
"""
Project A - 推論腳本
生成符合比賽格式的 CSV 檔案
輸出格式: crystal,lot,filename,class,cx,cy,w,h
"""

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm

def predict_and_generate_csv(
    model_path="/home/114078/projectA_work//models/best_model_yolov8m_correct.pt",
    test_root="/TOPIC/projectA/A_testing",  # 決賽時會是這個路徑
    output_path="/home/114078/submit/114078_projectA_ans.csv",
    conf_threshold=0.25
):
    """
    對測試集進行推論並生成 CSV 檔案

    參數:
        model_path: 訓練好的模型路徑
        test_root: 測試數據根目錄
        output_path: 輸出 CSV 路徑
        conf_threshold: 置信度閾   """

    print("="*80)
    print("Project A - 推論與 CSV 生成")
    print("="*80)

    # 檢查 GPU
    print(f"\nCUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 載入模型
    print(f"\n載入模型: {model_path}")
    if not Path(model_path).exists():
        print(f"錯誤：找不到模型檔案！")
        print(f"請確認路徑: {model_path}")
        return

    import os
    os.environ['YOLO_OFFLINE'] = '1'  # 設置離線模式
    model = YOLO(model_path, task='detect')  # 明確指定任務
    print("✓ 模型載入成功")

    # 檢查測試數據
    test_root = Path(test_root)
    if not test_root.exists():
        print(f"\n警告：測試數據路徑不存在: {test_root}")
        print("這是正常的，因為測試數據在決賽當天才會公布")
        print("請在決賽當天修改 test_root 路徑後執行")
        return

    print(f"\n測試數據路徑: {test_root}")

    # 準備結果列表
    results_list = []

    # 對三個晶向進行推論
    crystal_types = ["100", "111", "776"]

    total_images = 0
    total_predictions = 0

    for crystal in crystal_types:
        print(f"\n處理 {crystal} 晶向...")
        crystal_path = test_root / crystal

        if not crystal_path.exists():
            print(f"  警告：{crystal} 路徑不存在，跳過")
            continue

        # 獲取所有 lot
        lots = sorted([d for d in crystal_path.iterdir() if d.is_dir()])
        print(f"  找到 {len(lots)} 個 lots")

        for lot in tqdm(lots, desc=f"  {crystal}"):
            images_dir = lot / "images"

            if not images_dir.exists():
                continue

            # 獲取所有影像
            image_files = sorted(images_dir.glob("*.jpg"))

            for img_file in image_files:
                total_images += 1

                # 推論
                results = model.predict(
                    source=str(img_file),
                    conf=conf_threshold,
                    iou=0.45,
                    device=0,
                    verbose=False
                )

                result = results[0]

                # 如果有檢測到斷線
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        # 獲取標準化座標 (YOLO 格式)
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        conf = box.conf[0].item()

                        # 添加到結果列表
                        results_list.append({
                            'crystal': crystal,
                            'lot': lot.name,
                            'filename': img_file.name,
                            'class': 0,  # 固定為 0 (斷線類別)
                            'cx': f"{x_center:.6f}",
                            'cy': f"{y_center:.6f}",
                            'w': f"{width:.6f}",
                            'h': f"{height:.6f}"
                        })

                        total_predictions += 1

    print(f"\n" + "="*80)
    print(f"推論完成")
    print(f"="*80)
    print(f"總影像數: {total_images}")
    print(f"總預測框數: {total_predictions}")
    print(f"平均每張影像預測框數: {total_predictions/total_images if total_images > 0 else 0:.2f}")

    # 生成 CSV
    if len(results_list) == 0:
        print("\n警告：沒有任何預測結果！")
        print("可能原因：")
        print("1. 置信度閾值太高")
        print("2. 測試數據有問題")
        print("3. 模型效果不佳")
        return

    df = pd.DataFrame(results_list)

    # 確保輸出目錄存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存 CSV
    df.to_csv(output_path, index=False)

    print(f"\n✓ CSV 檔案已保存: {output_path}")
    print(f"\n前 10 筆預測結果:")
    print(df.head(10).to_string(index=False))

    # 統計每個晶向的預測數量
    print(f"\n各晶向預測統計:")
    for crystal in crystal_types:
        count = len(df[df['crystal'] == crystal])
        print(f"  {crystal}: {count} 個預測框")

    return df


def test_on_validation_set():
    """
    在驗證集上測試推論流程（用於決賽前驗證）
    """
    print("\n" + "="*80)
    print("在驗證集上測試推論流程")
    print("="*80)
        # 設置離線模式
    import os
    os.environ['YOLO_OFFLINE'] = '1'
    # 使用驗證集模擬測試
    val_root = Path("/home/114078/projectA_work/dataset/val/images")

    if not val_root.exists():
        print("錯誤：找不到驗證集")
        return

    # 簡化版測試：只處理前10張影像
    image_files = list(val_root.glob("*.jpg"))[:10]

    model_path = "/home/114078/projectA_work/models/best_model_yolov8m_correct.pt"
    # 檢查模型是否存在
    if not Path(model_path).exists():
        # 嘗試使用訓練結果目錄的模型
        alt_path = "/home/114078/projectA_work/results/yolov8n_breakline/weights/best.pt"
        if Path(alt_path).exists():
            print(f"使用訓練結果目錄的模型: {alt_path}")
            model_path = alt_path
        else:
            print(f"錯誤：找不到模型！")
            print(f"請檢查路徑:")
            print(f"  {model_path}")
            print(f"  {alt_path}")
            return

    print(f"載入模型: {model_path}")
    model = YOLO(model_path, task='detect')
    print("✓ 模型載入成功")


    results_list = []

    for img_file in tqdm(image_files, desc="測試推論"):
        results = model.predict(
            source=str(img_file),
            conf=0.25,
            iou=0.45,
            device=0,
            verbose=False
        )

        result = results[0]

        if len(result.boxes) > 0:
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()

                # 從檔名解析 crystal 和 lot
                # 檔名格式: crystal_原檔名
                parts = img_file.stem.split('_', 1)
                crystal = parts[0] if len(parts) > 0 else "unknown"

                results_list.append({
                    'crystal': crystal,
                    'lot': 'test_lot',
                    'filename': img_file.name,
                    'class': 0,
                    'cx': f"{x_center:.6f}",
                    'cy': f"{y_center:.6f}",
                    'w': f"{width:.6f}",
                    'h': f"{height:.6f}"
                })

    if len(results_list) > 0:
        df = pd.DataFrame(results_list)
        print(f"\n✓ 測試成功！生成 {len(results_list)} 個預測")
        print("\n預測結果範例:")
        print(df.head().to_string(index=False))
    else:
        print("\n⚠ 警告：沒有預測結果")
        print("可能原因：")
        print("1. 置信度閾值太高")
        print("2. 這10張影像剛好沒有斷線")
        print("3. 模型預測偏保守")

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 測試模式：在驗證集上測試
        print("執行測試模式...")
        test_on_validation_set()
    else:
        # 正式模式：對測試集推論
        print("執行正式推論...")
        predict_and_generate_csv()
