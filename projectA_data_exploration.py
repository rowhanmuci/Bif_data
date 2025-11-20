#!/usr/bin/env python3
"""
Project A - 矽單晶長晶斷線自動偵測
資料探勘與分析腳本

目的：
1. 統計各晶向的數據分布
2. 分析時序特性（影像間隔、斷線發生位置）
3. 分析標註框的尺寸分布
4. 檢查數據品質（缺失、異常）
5. 視覺化關鍵統計資訊
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import json
from datetime import datetime
import re

class ProjectADataExplorer:
    def __init__(self, data_root="/TOPIC/projectA/A_training_1"):
        self.data_root = Path(data_root)
        self.crystal_types = ["100", "111", "776"]
        self.stats = {}
        
    def explore_all(self):
        """執行完整的資料探勘流程"""
        print("="*80)
        print("Project A 資料探勘報告")
        print("="*80)
        
        # 1. 基本統計
        print("\n[1] 基本統計資訊")
        self.basic_statistics()
        
        # 2. 時序分析
        print("\n[2] 時序特性分析")
        self.temporal_analysis()
        
        # 3. 標註框分析
        print("\n[3] 標註框尺寸分析")
        self.bbox_analysis()
        
        # 4. 數據品質檢查
        print("\n[4] 數據品質檢查")
        self.quality_check()
        
        # 5. 斷線發生位置分析
        print("\n[5] 斷線在序列中的位置分析")
        self.breakline_position_analysis()
        
        # 6. 保存統計結果
        self.save_statistics()
        
    def basic_statistics(self):
        """基本統計資訊"""
        total_stats = {
            'crystal_type': [],
            'num_lots': [],
            'total_images': [],
            'total_labels': [],
            'breakline_ratio': []
        }
        
        for crystal in self.crystal_types:
            crystal_path = self.data_root / crystal
            
            if not crystal_path.exists():
                print(f"警告：{crystal} 晶向資料夾不存在")
                continue
            
            # 統計 lots
            lots = [d for d in crystal_path.iterdir() if d.is_dir()]
            num_lots = len(lots)
            
            # 統計影像和標註
            total_images = 0
            total_labels = 0
            
            for lot in lots:
                images_dir = lot / "images"
                labels_dir = lot / "labels"
                
                if images_dir.exists():
                    total_images += len(list(images_dir.glob("*.jpg")))
                if labels_dir.exists():
                    total_labels += len(list(labels_dir.glob("*.txt")))
            
            breakline_ratio = total_labels / total_images if total_images > 0 else 0
            
            total_stats['crystal_type'].append(crystal)
            total_stats['num_lots'].append(num_lots)
            total_stats['total_images'].append(total_images)
            total_stats['total_labels'].append(total_labels)
            total_stats['breakline_ratio'].append(breakline_ratio)
            
            print(f"\n{crystal} 晶向:")
            print(f"  - Lots 數量: {num_lots}")
            print(f"  - 影像總數: {total_images}")
            print(f"  - 標註總數: {total_labels}")
            print(f"  - 斷線比例: {breakline_ratio:.2%}")
            print(f"  - 平均每個 lot: {total_images/num_lots:.1f} 張影像, {total_labels/num_lots:.1f} 個斷線")
        
        # 保存統計
        self.stats['basic'] = pd.DataFrame(total_stats)
        
        # 總計
        print(f"\n總計:")
        print(f"  - 總 Lots: {sum(total_stats['num_lots'])}")
        print(f"  - 總影像數: {sum(total_stats['total_images'])}")
        print(f"  - 總標註數: {sum(total_stats['total_labels'])}")
        
    def temporal_analysis(self):
        """時序特性分析"""
        print("\n分析影像時間間隔和斷線發生時序...")
        
        temporal_stats = {
            'crystal_type': [],
            'avg_time_interval': [],
            'std_time_interval': [],
            'median_time_interval': [],
            'first_breakline_position': []
        }
        
        for crystal in self.crystal_types:
            crystal_path = self.data_root / crystal
            lots = [d for d in crystal_path.iterdir() if d.is_dir()]
            
            time_intervals = []
            first_breakline_positions = []
            
            for lot in lots[:10]:  # 取前10個lot作為樣本
                images_dir = lot / "images"
                labels_dir = lot / "labels"
                
                if not images_dir.exists():
                    continue
                
                # 獲取影像檔名並排序
                image_files = sorted(images_dir.glob("*.jpg"))
                
                if len(image_files) < 2:
                    continue
                
                # 計算時間間隔
                timestamps = []
                for img_file in image_files:
                    # 解析檔名中的時間：20240602_093811_CCD.jpg
                    match = re.search(r'(\d{8}_\d{6})', img_file.name)
                    if match:
                        time_str = match.group(1)
                        try:
                            dt = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                            timestamps.append(dt)
                        except:
                            pass
                
                # 計算間隔（秒）
                if len(timestamps) >= 2:
                    intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                                for i in range(len(timestamps)-1)]
                    time_intervals.extend(intervals)
                
                # 找第一個斷線的位置
                label_files = sorted(labels_dir.glob("*.txt"))
                if len(label_files) > 0 and len(image_files) > 0:
                    first_label = label_files[0].stem  # 去掉 .txt
                    
                    # 找這個檔案在影像序列中的位置
                    for idx, img_file in enumerate(image_files):
                        if img_file.stem == first_label:
                            first_breakline_positions.append(idx / len(image_files))
                            break
            
            # 統計
            if len(time_intervals) > 0:
                avg_interval = np.mean(time_intervals)
                std_interval = np.std(time_intervals)
                median_interval = np.median(time_intervals)
            else:
                avg_interval = std_interval = median_interval = 0
            
            avg_first_position = np.mean(first_breakline_positions) if first_breakline_positions else 0
            
            temporal_stats['crystal_type'].append(crystal)
            temporal_stats['avg_time_interval'].append(avg_interval)
            temporal_stats['std_time_interval'].append(std_interval)
            temporal_stats['median_time_interval'].append(median_interval)
            temporal_stats['first_breakline_position'].append(avg_first_position)
            
            print(f"\n{crystal} 晶向:")
            print(f"  - 平均影像間隔: {avg_interval:.1f} 秒")
            print(f"  - 間隔標準差: {std_interval:.1f} 秒")
            print(f"  - 中位數間隔: {median_interval:.1f} 秒")
            print(f"  - 第一個斷線平均位置: {avg_first_position:.1%} (在序列中)")
        
        self.stats['temporal'] = pd.DataFrame(temporal_stats)
    
    def bbox_analysis(self):
        """標註框尺寸分析"""
        print("\n分析標註框尺寸分布...")
        
        bbox_data = {
            'crystal_type': [],
            'x_center': [],
            'y_center': [],
            'width': [],
            'height': []
        }
        
        for crystal in self.crystal_types:
            crystal_path = self.data_root / crystal
            lots = [d for d in crystal_path.iterdir() if d.is_dir()]
            
            x_centers, y_centers, widths, heights = [], [], [], []
            
            for lot in lots:
                labels_dir = lot / "labels"
                
                if not labels_dir.exists():
                    continue
                
                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls, x, y, w, h = map(float, parts)
                                x_centers.append(x)
                                y_centers.append(y)
                                widths.append(w)
                                heights.append(h)
            
            print(f"\n{crystal} 晶向 (共 {len(widths)} 個標註框):")
            if len(widths) > 0:
                print(f"  寬度 (normalized):")
                print(f"    - 平均: {np.mean(widths):.4f}, 標準差: {np.std(widths):.4f}")
                print(f"    - 最小: {np.min(widths):.4f}, 最大: {np.max(widths):.4f}")
                print(f"  高度 (normalized):")
                print(f"    - 平均: {np.mean(heights):.4f}, 標準差: {np.std(heights):.4f}")
                print(f"    - 最小: {np.min(heights):.4f}, 最大: {np.max(heights):.4f}")
                print(f"  中心位置 (y):")
                print(f"    - 平均: {np.mean(y_centers):.4f}, 標準差: {np.std(y_centers):.4f}")
                
                bbox_data['crystal_type'].extend([crystal] * len(widths))
                bbox_data['x_center'].extend(x_centers)
                bbox_data['y_center'].extend(y_centers)
                bbox_data['width'].extend(widths)
                bbox_data['height'].extend(heights)
        
        self.stats['bbox'] = pd.DataFrame(bbox_data)
    
    def quality_check(self):
        """數據品質檢查"""
        print("\n檢查數據品質問題...")
        
        issues = []
        
        for crystal in self.crystal_types:
            crystal_path = self.data_root / crystal
            lots = [d for d in crystal_path.iterdir() if d.is_dir()]
            
            for lot in lots:
                images_dir = lot / "images"
                labels_dir = lot / "labels"
                
                # 檢查資料夾是否存在
                if not images_dir.exists():
                    issues.append(f"{crystal}/{lot.name}: 缺少 images 資料夾")
                    continue
                    
                if not labels_dir.exists():
                    issues.append(f"{crystal}/{lot.name}: 缺少 labels 資料夾")
                    continue
                
                # 檢查是否有影像但無標註
                image_files = list(images_dir.glob("*.jpg"))
                label_files = list(labels_dir.glob("*.txt"))
                
                if len(image_files) == 0:
                    issues.append(f"{crystal}/{lot.name}: 沒有影像檔案")
                
                # 檢查標註檔案格式
                for label_file in label_files[:5]:  # 抽查前5個
                    with open(label_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            parts = line.strip().split()
                            if len(parts) != 5:
                                issues.append(
                                    f"{crystal}/{lot.name}/{label_file.name}:{line_num} "
                                    f"格式錯誤，應該有5個值"
                                )
        
        if len(issues) == 0:
            print("✓ 沒有發現數據品質問題")
        else:
            print(f"✗ 發現 {len(issues)} 個問題:")
            for issue in issues[:10]:  # 只顯示前10個
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... 還有 {len(issues)-10} 個問題")
        
        self.stats['quality_issues'] = issues
    
    def breakline_position_analysis(self):
        """分析斷線在序列中發生的位置"""
        print("\n分析斷線在影像序列中的分布...")
        
        position_data = {
            'crystal_type': [],
            'lot_name': [],
            'total_images': [],
            'num_breaklines': [],
            'first_breakline_position': [],
            'last_breakline_position': []
        }
        
        for crystal in self.crystal_types:
            crystal_path = self.data_root / crystal
            lots = [d for d in crystal_path.iterdir() if d.is_dir()]
            
            for lot in lots:
                images_dir = lot / "images"
                labels_dir = lot / "labels"
                
                if not images_dir.exists() or not labels_dir.exists():
                    continue
                
                image_files = sorted(images_dir.glob("*.jpg"))
                label_files = sorted(labels_dir.glob("*.txt"))
                
                if len(image_files) == 0:
                    continue
                
                # 找到有標註的影像索引
                image_names = [img.stem for img in image_files]
                label_names = [lbl.stem for lbl in label_files]
                
                breakline_indices = []
                for label_name in label_names:
                    if label_name in image_names:
                        idx = image_names.index(label_name)
                        breakline_indices.append(idx)
                
                if len(breakline_indices) > 0:
                    position_data['crystal_type'].append(crystal)
                    position_data['lot_name'].append(lot.name)
                    position_data['total_images'].append(len(image_files))
                    position_data['num_breaklines'].append(len(breakline_indices))
                    position_data['first_breakline_position'].append(
                        breakline_indices[0] / len(image_files)
                    )
                    position_data['last_breakline_position'].append(
                        breakline_indices[-1] / len(image_files)
                    )
        
        df = pd.DataFrame(position_data)
        
        for crystal in self.crystal_types:
            crystal_data = df[df['crystal_type'] == crystal]
            if len(crystal_data) > 0:
                print(f"\n{crystal} 晶向:")
                print(f"  - 平均每個 lot 影像數: {crystal_data['total_images'].mean():.1f}")
                print(f"  - 平均每個 lot 斷線數: {crystal_data['num_breaklines'].mean():.1f}")
                print(f"  - 第一個斷線平均位置: {crystal_data['first_breakline_position'].mean():.1%}")
                print(f"  - 最後一個斷線平均位置: {crystal_data['last_breakline_position'].mean():.1%}")
        
        self.stats['position'] = df
    
    def save_statistics(self):
        """保存統計結果"""
        output_dir = Path("/home/114078/projectA_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # 保存各項統計為 CSV
        for name, df in self.stats.items():
            if isinstance(df, pd.DataFrame):
                output_file = output_dir / f"{name}_stats.csv"
                df.to_csv(output_file, index=False)
                print(f"\n已保存: {output_file}")
        
        # 保存摘要為 JSON
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_root': str(self.data_root),
            'crystal_types': self.crystal_types
        }
        
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n統計結果已保存至: {output_dir}")


if __name__ == "__main__":
    explorer = ProjectADataExplorer()
    explorer.explore_all()
    
    print("\n" + "="*80)
    print("資料探勘完成！")
    print("="*80)
