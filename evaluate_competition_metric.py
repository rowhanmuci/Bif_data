
#!/usr/bin/env python3
"""
比賽官方評分指標計算
包含 delay_penalty 的 Weighted F1 Score
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def calculate_iou(box1, box2):
    """
    計算兩個框的 IoU
    box: [x_center, y_center, width, height] (normalized)
    """
    # 轉換為 [x1, y1, x2, y2]
    def xywh_to_xyxy(box):
        x_center, y_center, w, h = box
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return [x1, y1, x2, y2]

    box1_xyxy = xywh_to_xyxy(box1)
    box2_xyxy = xywh_to_xyxy(box2)

    # 計算交集
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # 計算聯集
    box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])

    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.3):
    """
    將預測框與 GT 框配對（IoU >= threshold）
    返回: TP, FP, FN
    """
    matched_gt = set()
    tp = 0
    fp = 0

    for pred_box in pred_boxes:
        matched = False
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            iou = calculate_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                matched = True
                break

        if not matched:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


def calculate_f1_for_sequence(tp, fp, fn):
    """
    計算單一序列的 F1 Score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall


def calculate_delay_penalty(t_detect, t0):
    """
    計算延遲懲罰
    W(Tdetect, T0) = exp(-0.01 * |Tdetect - T0|)
    """
    if t_detect is None:
        return 0  # 完全沒檢測到

    delay = abs(t_detect - t0)
    penalty = np.exp(-0.01 * delay)

    return penalty


def evaluate_with_competition_metric(
    model_path,
    val_data_root="/home/114078/projectA_work/dataset/val",
    conf_threshold=0.25
):
    """
    使用比賽官方評分標準評估模型
    """
    # 設置離線模式
    os.environ['YOLO_OFFLINE'] = '1'

    print("="*80)
    print("比賽官方評分指標計算")
    print("="*80)

    # 載入模型
    print(f"\n載入模型: {model_path}")
    model = YOLO(model_path)
    print("✓ 模型載入成功")

    # 讀取驗證集
    val_images_dir = Path(val_data_root) / "images"
    val_labels_dir = Path(val_data_root) / "labels"

    if not val_images_dir.exists():
        print(f"錯誤：找不到驗證集影像目錄 {val_images_dir}")
        return

    # 組織數據：按照 lot 分組
    # 由於驗證集已經扁平化，我們需要模擬序列結構
    # 實際比賽中會有完整的 lot 結構

    print(f"\n警告：驗證集已扁平化，無法完全模擬序列結構")
    print(f"這裡我們按照檔名排序來模擬序列")

    # 獲取所有影像
    image_files = sorted(val_images_dir.glob("*.jpg"))

    print(f"\n驗證集影像數: {len(image_files)}")

    # 簡化版：將每個晶向視為一個序列
    sequences = defaultdict(list)

    for img_file in image_files:
        # 從檔名提取晶向 (crystal_...)
        parts = img_file.stem.split('_', 1)
        crystal = parts[0] if len(parts) > 0 else "unknown"

        sequences[crystal].append(img_file)

    print(f"\n按晶向分組:")
    for crystal, files in sequences.items():
        print(f"  {crystal}: {len(files)} 張影像")

    # 對每個序列進行評估
    all_weighted_f1 = []
    detailed_results = []

    for crystal, seq_images in sequences.items():
        print(f"\n評估 {crystal} 晶向序列...")

        # 按檔名排序（模擬時間順序）
        seq_images = sorted(seq_images)

        # 收集 GT 和預測
        gt_info = []  # (frame_idx, boxes)
        pred_info = []  # (frame_idx, boxes)

        for frame_idx, img_file in enumerate(tqdm(seq_images, desc=f"  {crystal}")):
            # 讀取 GT
            label_file = val_labels_dir / f"{img_file.stem}.txt"
            gt_boxes = []

            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, x, y, w, h = map(float, parts)
                            gt_boxes.append([x, y, w, h])

            if len(gt_boxes) > 0:
                gt_info.append((frame_idx, gt_boxes))

            # 預測
            results = model.predict(
                source=str(img_file),
                conf=conf_threshold,
                iou=0.45,
                device=0,
                verbose=False
            )

            result = results[0]
            pred_boxes = []

            if len(result.boxes) > 0:
                for box in result.boxes:
                    x, y, w, h = box.xywhn[0].tolist()
                    pred_boxes.append([x, y, w, h])

            if len(pred_boxes) > 0:
                pred_info.append((frame_idx, pred_boxes))

        # 計算這個序列的指標
        if len(gt_info) == 0:
            print(f"  警告：{crystal} 序列沒有 GT 標註，跳過")
            continue

        # 找第一個斷線的幀
        t0 = gt_info[0][0]  # GT 第一個斷線

        # 找預測第一個斷線的幀
        t_detect = pred_info[0][0] if len(pred_info) > 0 else None

        # 計算整個序列的 TP, FP, FN
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # 將 GT 和預測對齊到每一幀
        gt_dict = {idx: boxes for idx, boxes in gt_info}
        pred_dict = {idx: boxes for idx, boxes in pred_info}

        all_frames = set(gt_dict.keys()) | set(pred_dict.keys())

        for frame_idx in all_frames:
            gt_boxes = gt_dict.get(frame_idx, [])
            pred_boxes = pred_dict.get(frame_idx, [])

            tp, fp, fn = match_predictions_to_gt(pred_boxes, gt_boxes, iou_threshold=0.3)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # 計算 F1
        f1, precision, recall = calculate_f1_for_sequence(total_tp, total_fp, total_fn)

        # 計算 delay penalty
        delay_penalty = calculate_delay_penalty(t_detect, t0)

        # 計算 weighted F1
        weighted_f1 = f1 * delay_penalty

        all_weighted_f1.append(weighted_f1)

        # 記錄詳細結果
        detailed_results.append({
            'crystal': crystal,
            'num_images': len(seq_images),
            'num_gt_frames': len(gt_info),
            'num_pred_frames': len(pred_info),
            't0': t0,
            't_detect': t_detect,
            'delay': abs(t_detect - t0) if t_detect is not None else None,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'delay_penalty': delay_penalty,
            'weighted_f1': weighted_f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        })

        print(f"\n  {crystal} 序列結果:")
        print(f"    影像數: {len(seq_images)}")
        print(f"    GT 斷線幀數: {len(gt_info)}")
        print(f"    預測斷線幀數: {len(pred_info)}")
        print(f"    第一個斷線 (GT): 幀 {t0}")
        print(f"    第一個斷線 (預測): 幀 {t_detect if t_detect is not None else 'N/A'}")
        print(f"    延遲: {abs(t_detect - t0) if t_detect is not None else 'N/A'} 幀")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    Delay Penalty: {delay_penalty:.4f}")
        print(f"    Weighted F1: {weighted_f1:.4f}")

    # 計算最終分數
    if len(all_weighted_f1) > 0:
        final_score = np.mean(all_weighted_f1)
    else:
        final_score = 0

    print("\n" + "="*80)
    print("最終評分結果")
    print("="*80)

    print(f"\nAverage Weighted F1 Score: {final_score:.4f} ({final_score*100:.2f}%)")

    print(f"\n各序列詳細結果:")
    print(f"{'晶向':<10} {'F1':<8} {'Delay':<8} {'Penalty':<10} {'Weighted F1':<12}")
    print("-"*60)
    for r in detailed_results:
        delay_str = f"{r['delay']}" if r['delay'] is not None else "N/A"
        print(f"{r['crystal']:<10} {r['f1']:<8.4f} {delay_str:<8} {r['delay_penalty']:<10.4f} {r['weighted_f1']:<12.4f}")

    # 與比賽要求比較
    print(f"\n" + "="*80)
    print("與比賽要求比較")
    print("="*80)
    print(f"比賽要求: Accuracy > 85%")
    print(f"當前 Weighted F1: {final_score*100:.2f}%")

    if final_score > 0.85:
        print(f"✓ 符合比賽標準！")
    else:
        print(f"⚠ 需要改進（差距: {(0.85-final_score)*100:.2f}%）")

    # 保存結果
    results_file = Path("/home/114078/projectA_work/competition_evaluation.txt")
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("比賽官方評分結果\n")
        f.write("="*80 + "\n\n")
        f.write(f"Average Weighted F1 Score: {final_score:.4f} ({final_score*100:.2f}%)\n\n")
        f.write("各序列詳細結果:\n")
        f.write("-"*80 + "\n")
        for r in detailed_results:
            f.write(f"\n{r['crystal']} 晶向:\n")
            f.write(f"  影像數: {r['num_images']}\n")
            f.write(f"  F1 Score: {r['f1']:.4f}\n")
            f.write(f"  Precision: {r['precision']:.4f}\n")
            f.write(f"  Recall: {r['recall']:.4f}\n")
            f.write(f"  Delay: {r['delay']} 幀\n")
            f.write(f"  Delay Penalty: {r['delay_penalty']:.4f}\n")
            f.write(f"  Weighted F1: {r['weighted_f1']:.4f}\n")

    print(f"\n✓ 評估結果已保存至: {results_file}")

    return final_score, detailed_results


if __name__ == "__main__":
    model_path = "/home/114078/projectA_work/results/yolov8n_breakline/weights/best.pt"

    if not Path(model_path).exists():
        print(f"錯誤：找不到模型 {model_path}")
        exit(1)

    final_score, details = evaluate_with_competition_metric(
        model_path=model_path,
        conf_threshold=0.25
    )

    print("\n" + "="*80)
    print("評估完成")
    print("="*80)
