# -*- coding: utf-8 -*-
"""설정 관리 모듈"""

from pathlib import Path


class Config:
    """프로젝트 전역 설정"""
    
    # 기본 경로
    PROJECT_ROOT = Path.cwd()
    WORKSPACE_ROOT = PROJECT_ROOT / 'workspace'
    PROJECT_DATA_ROOT = PROJECT_ROOT / 'PCB_defect_detection_Project'
    
    # 데이터 경로
    RAW_DATA_PATH = WORKSPACE_ROOT / 'raw_data'
    IMAGES_DIR = RAW_DATA_PATH / 'images'
    ANNOTATIONS_DIR = RAW_DATA_PATH / 'Annotations'
    RESIZED_IMAGES_DIR = WORKSPACE_ROOT / 'images_resized'
    PROCESSED_DATA_PATH = WORKSPACE_ROOT / 'data_processed'
    
    # 결과 경로
    RESULTS_DIR = PROJECT_DATA_ROOT / 'results'
    YAML_PATH = PROJECT_DATA_ROOT / 'data.yaml'
    
    # Kaggle 데이터셋
    KAGGLE_DATASET_NAME = 'akhatova/pcb-defects'
    
    # 클래스 정의
    CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 
               'short', 'spur', 'spurious_copper']
    
    # 이미지 설정
    IMAGE_SIZE = 640
    TARGET_SIZE = (IMAGE_SIZE, IMAGE_SIZE)
    
    # 데이터 분할 비율
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # 학습 하이퍼파라미터
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    OPTIMIZER = 'Adam'
    
    # 데이터 증강 설정
    HSV_H = 0.015
    HSV_S = 0.7
    HSV_V = 0.4
    DEGREES = 10.0
    FLIP_LR = 0.0
    MIXUP = 0.3
    
    # YOLO 모델
    MODEL_NAME = 'yolo11s.pt'
    RUN_NAME = 'yolo11s_run'
    
    # 시드 (재현성)
    RANDOM_SEED = 42
    
    @classmethod
    def setup_directories(cls):
        """필요한 디렉토리 생성"""
        cls.WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        cls.PROJECT_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            (cls.PROCESSED_DATA_PATH / 'images' / split).mkdir(parents=True, exist_ok=True)
            (cls.PROCESSED_DATA_PATH / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """설정 정보 출력"""
        print("=" * 60)
        print("프로젝트 설정")
        print("=" * 60)
        print(f"프로젝트 루트: {cls.PROJECT_ROOT}")
        print(f"작업 공간: {cls.WORKSPACE_ROOT}")
        print(f"이미지 크기: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"클래스 수: {len(cls.CLASSES)}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print("=" * 60)