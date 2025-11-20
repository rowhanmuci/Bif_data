#!/usr/bin/env python3
"""
在終端機直接檢查斷線特徵
"""

from pathlib import Path
import cv2
import numpy as np

def analyze_image_characteristics(image_path, label_path=None):
    """分析單張影像的特徵"""
    
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    height, width = img.shape[:2]
    
    # 基本統計
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # 邊緣檢測（粗略估計紋理複雜度）
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    
    info = {
        'size': f"{width}x{height}",
        'brightness': f"{mean_brightness:.1f}±{std_brightness:.1f}",
        'edge_density': f"{edge_density:.4f}",
    }
    
    # 如果有標註，分析標註區域
    if label_path and label_path.exists():
        with open(label_path, 'r') as f:
            bboxes = []
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, x_c, y_c, w, h = map(float, parts)
                    bboxes.append((x_c, y_c, w, h))
            
            info['num_bboxes'] = len(bboxes)
            
            # 分析標註區域的特徵
            if len(bboxes) > 0:
                bbox_analysis = []
                for x_c, y_c, w, h in bboxes:
                    # 提取 ROI
                    x1 = int((x_c - w/2) * width)
                    y1 = int((y_c - h/2) * height)
                    x2 = int((x_c + w/2) * width)
                    y2 = int((y_c + h/2) * height)
                    
                    # 確保座標在範圍內
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    roi = gray[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        roi_brightness = np.mean(roi)
                        roi_std = np.std(roi)
                        bbox_analysis.append(f"亮度={roi_brightness:.1f}±{roi_std:.1f}")
                
                info['bbox_features'] = bbox_analysis
    
    return info


def compare_normal_vs_breakline(data_root="/TOPIC/projectA/A_training_1", crystal="100"):
    """比較正常影像 vs 斷線影像的特徵"""
    
    print(f"\n{'='*80}")
    print(f"分析 {crystal} 晶向 - 正常 vs 斷線影像特徵比較")
    print('='*80)
    
    crystal_path = Path(data_root) / crystal
    lots = sorted([d for d in crystal_path.iterdir() if d.is_dir()])
    
    # 找一個有斷線的 lot
    for lot in lots:
        images_dir = lot / "images"
        labels_dir = lot / "labels"
        
        image_files = sorted(images_dir.glob("*.jpg"))
        label_files = {lbl.stem: lbl for lbl in labels_dir.glob("*.txt") 
                      if lbl.name != "classes.txt"}
        
        if len(label_files) == 0:
            continue
        
        # 找到第一個斷線的位置
        first_breakline_idx = None
        for idx, img_file in enumerate(image_files):
            if img_file.stem in label_files:
                first_breakline_idx = idx
                break
        
        if first_breakline_idx is None or first_breakline_idx < 5:
            continue
        
        print(f"\nLot: {lot.name}")
        print(f"總影像: {len(image_files)}, 斷線影像: {len(label_files)}")
        print(f"第一個斷線位置: {first_breakline_idx}/{len(image_files)}")
        
        # 比較：斷線前5幀、斷線前1幀、第1個斷線、第5個斷線
        frames_to_check = [
            (first_breakline_idx - 5, "斷線前5幀 (正常)"),
            (first_breakline_idx - 1, "斷線前1幀 (正常)"),
            (first_breakline_idx, "第1個斷線"),
        ]
        
        # 找第5個斷線
        breakline_count = 0
        for idx, img_file in enumerate(image_files[first_breakline_idx:], start=first_breakline_idx):
            if img_file.stem in label_files:
                breakline_count += 1
                if breakline_count == 5:
                    frames_to_check.append((idx, "第5個斷線"))
                    break
        
        print(f"\n{'位置':<20} {'影像尺寸':<15} {'亮度':<20} {'邊緣密度':<12} {'標註數':<8}")
        print('-'*80)
        
        for frame_idx, description in frames_to_check:
            if frame_idx < 0 or frame_idx >= len(image_files):
                continue
            
            img_file = image_files[frame_idx]
            label_file = label_files.get(img_file.stem)
            
            info = analyze_image_characteristics(img_file, label_file)
            
            if info:
                num_bboxes = info.get('num_bboxes', 0)
                print(f"{description:<20} {info['size']:<15} {info['brightness']:<20} "
                      f"{info['edge_density']:<12} {num_bboxes:<8}")
                
                # 如果有標註，顯示標註區域特徵
                if 'bbox_features' in info:
                    for bbox_feat in info['bbox_features']:
                        print(f"  → 標註區域: {bbox_feat}")
        
        # 只分析第一個找到的 lot
        break
    
    print()


def main():
    """主程式"""
    print("\n" + "="*80)
    print("斷線特徵分析")
    print("="*80)
    
    for crystal in ["100", "111", "776"]:
        compare_normal_vs_breakline(crystal=crystal)
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()