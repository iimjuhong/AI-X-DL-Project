# -*- coding: utf-8 -*-
"""데이터셋 분할 모듈"""

import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import Config


class DatasetSplitter:
    """데이터셋 분할 및 YOLO 라벨 생성"""
    
    def __init__(self, annotations_df):
        self.config = Config
        self.annotations_df = annotations_df
        self.yolo_df = None
        self.splits = {}
    
    def convert_to_yolo_format(self):
        """YOLO 형식으로 변환"""
        print("\n[단계] YOLO 형식 변환")
        
        self.yolo_df = self.annotations_df.copy()
        
        # 클래스 ID 매핑
        self.yolo_df['class_id'] = self.yolo_df['class'].apply(
            lambda x: self.config.CLASSES.index(x.lower())
        )
        
        # YOLO 형식 좌표 계산 (정규화된 중심점, 너비, 높이)
        self.yolo_df['x_center'] = ((self.yolo_df['xmin'] + self.yolo_df['xmax']) / 2) / self.yolo_df['width']
        self.yolo_df['y_center'] = ((self.yolo_df['ymin'] + self.yolo_df['ymax']) / 2) / self.yolo_df['height']
        self.yolo_df['bbox_w'] = (self.yolo_df['xmax'] - self.yolo_df['xmin']) / self.yolo_df['width']
        self.yolo_df['bbox_h'] = (self.yolo_df['ymax'] - self.yolo_df['ymin']) / self.yolo_df['height']
        
        print("✓ YOLO 형식 변환 완료")
        return self.yolo_df
    
    def split_dataset(self):
        """데이터셋을 train/val/test로 분할"""
        print("\n[단계] 데이터셋 분할")
        
        unique_filenames = self.yolo_df['filename'].unique()
        np.random.seed(self.config.RANDOM_SEED)
        np.random.shuffle(unique_filenames)
        
        train_end = int(len(unique_filenames) * self.config.TRAIN_RATIO)
        val_end = train_end + int(len(unique_filenames) * self.config.VAL_RATIO)
        
        self.splits = {
            'train': unique_filenames[:train_end],
            'val': unique_filenames[train_end:val_end],
            'test': unique_filenames[val_end:]
        }
        
        print(f"Train: {len(self.splits['train'])}장, "
              f"Val: {len(self.splits['val'])}장, "
              f"Test: {len(self.splits['test'])}장")
        
        return self.splits
    
    def save_split_data(self):
        """분할된 데이터 저장 (이미지 + 라벨)"""
        print("\n[단계] 파일 저장 중...")
        
        for split, filenames in self.splits.items():
            for fname in tqdm(filenames, desc=f"{split} 처리"):
                # 이미지 복사
                src_img = self.config.RESIZED_IMAGES_DIR / fname
                dst_img = self.config.PROCESSED_DATA_PATH / 'images' / split / fname
                
                if src_img.exists():
                    shutil.copy(src_img, dst_img)
                
                # 라벨 생성
                self._create_label_file(fname, split)
        
        print("✓ 데이터셋 저장 완료")
    
    def _create_label_file(self, filename, split):
        """YOLO 형식 라벨 파일 생성"""
        file_objects = self.yolo_df[self.yolo_df['filename'] == filename]
        label_path = self.config.PROCESSED_DATA_PATH / 'labels' / split / f"{Path(filename).stem}.txt"
        
        with open(label_path, 'w') as f:
            for _, row in file_objects.iterrows():
                f.write(f"{int(row['class_id'])} "
                       f"{row['x_center']:.6f} {row['y_center']:.6f} "
                       f"{row['bbox_w']:.6f} {row['bbox_h']:.6f}\n")
    
    def create_yaml_file(self):
        """data.yaml 파일 생성"""
        print("\n[단계] data.yaml 파일 생성")
        
        dataset_root = self.config.PROCESSED_DATA_PATH.absolute()
        
        yaml_content = f"""path: {dataset_root.as_posix()}
train: images/train
val: images/val
test: images/test

names:
"""
        for i, class_name in enumerate(self.config.CLASSES):
            yaml_content += f"  {i}: {class_name}\n"
        
        with open(self.config.YAML_PATH, 'w') as f:
            f.write(yaml_content)
        
        print(f"✓ data.yaml 저장 완료: {self.config.YAML_PATH}")