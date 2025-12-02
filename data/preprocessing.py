# -*- coding: utf-8 -*-
"""데이터 전처리 모듈"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from config import Config


class DataPreprocessor:
    """데이터 전처리 (XML 파싱, 이미지 리사이즈)"""
    
    def __init__(self):
        self.config = Config
        self.annotations_df = None
        self.resized_annotations_df = None
    
    def parse_xml_annotations(self):
        """XML 어노테이션 파일 파싱"""
        print("\n[단계] XML 어노테이션 파싱")
        
        xml_files = list(self.config.ANNOTATIONS_DIR.rglob('*.xml'))
        print(f"발견된 XML 파일 수: {len(xml_files)}")
        
        df_list = []
        for xml_file in tqdm(xml_files, desc="XML 파싱"):
            df_list.append(pd.DataFrame(self._parse_single_xml(xml_file)))
        
        self.annotations_df = pd.concat(df_list, ignore_index=True)
        print(f"✓ 총 {len(self.annotations_df)}개의 객체 파싱 완료")
        
        return self.annotations_df
    
    def _parse_single_xml(self, xml_file):
        """단일 XML 파일 파싱"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        data = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            
            data.append({
                'filename': filename,
                'width': width,
                'height': height,
                'class': name,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
        
        return data
    
    def resize_images(self):
        """이미지 리사이즈 (병렬 처리)"""
        print("\n[단계] 이미지 리사이즈 및 좌표 변환")
        
        self.config.RESIZED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        
        unique_files = self.annotations_df[['filename', 'width', 'height']].drop_duplicates()
        
        resize_tasks = [
            (row, self.config.IMAGES_DIR, self.config.RESIZED_IMAGES_DIR, self.config.TARGET_SIZE)
            for _, row in unique_files.iterrows()
        ]
        
        print(f"이미지 리사이즈 작업 시작 ({len(resize_tasks)}장)...")
        
        with ProcessPoolExecutor() as executor:
            list(tqdm(
                executor.map(self._process_single_image, resize_tasks),
                total=len(resize_tasks),
                desc="이미지 리사이즈"
            ))
        
        # 좌표 스케일링
        self._scale_coordinates()
        
        print("✓ 이미지 처리 완료")
        
        return self.resized_annotations_df
    
    @staticmethod
    def _process_single_image(args):
        """단일 이미지 리사이즈 (병렬 처리용)"""
        row, input_dir, output_dir, target_size = args
        
        try:
            image_path = list(input_dir.rglob(row['filename']))[0]
            image = cv2.imread(str(image_path))
            
            if image is None:
                return None
            
            resized_image = cv2.resize(image, target_size)
            output_path = output_dir / row['filename']
            cv2.imwrite(str(output_path), resized_image)
            
            return True
            
        except Exception as e:
            print(f"⚠ 이미지 처리 실패: {row['filename']} - {e}")
            return None
    
    def _scale_coordinates(self):
        """바운딩 박스 좌표 스케일링"""
        self.resized_annotations_df = self.annotations_df.copy()
        
        scale_w = self.config.IMAGE_SIZE / self.resized_annotations_df['width']
        scale_h = self.config.IMAGE_SIZE / self.resized_annotations_df['height']
        
        self.resized_annotations_df['xmin'] = (self.resized_annotations_df['xmin'] * scale_w).astype(int)
        self.resized_annotations_df['xmax'] = (self.resized_annotations_df['xmax'] * scale_w).astype(int)
        self.resized_annotations_df['ymin'] = (self.resized_annotations_df['ymin'] * scale_h).astype(int)
        self.resized_annotations_df['ymax'] = (self.resized_annotations_df['ymax'] * scale_h).astype(int)
        self.resized_annotations_df['width'] = self.config.IMAGE_SIZE
        self.resized_annotations_df['height'] = self.config.IMAGE_SIZE