# -*- coding: utf-8 -*-
"""모델 학습 모듈"""

from pathlib import Path
from config import Config


class ModelTrainer:
    """YOLO 모델 학습"""
    
    def __init__(self):
        self.config = Config
        self.model = None
        self.results = None
    
    def train(self):
        """모델 학습 실행"""
        print("\n" + "=" * 60)
        print("YOLO 모델 학습")
        print("=" * 60)
        
        try:
            from ultralytics import YOLO
            
            print(f"모델 로드: {self.config.MODEL_NAME}")
            self.model = YOLO(self.config.MODEL_NAME)
            
            print("\n학습 시작...")
            self.results = self.model.train(
                data=str(self.config.YAML_PATH),
                epochs=self.config.EPOCHS,
                batch=self.config.BATCH_SIZE,
                imgsz=self.config.IMAGE_SIZE,
                project=str(self.config.RESULTS_DIR),
                name=self.config.RUN_NAME,
                exist_ok=True,
                optimizer=self.config.OPTIMIZER,
                hsv_h=self.config.HSV_H,
                hsv_s=self.config.HSV_S,
                hsv_v=self.config.HSV_V,
                degrees=self.config.DEGREES,
                fliplr=self.config.FLIP_LR,
                mixup=self.config.MIXUP
            )
            
            results_path = self.config.RESULTS_DIR / self.config.RUN_NAME
            print(f"\n✓ 학습 완료! 결과 위치: {results_path}")
            
            return self.model, self.results
            
        except ImportError:
            print("\n❌ ultralytics가 설치되지 않았습니다.")
            print("설치 명령어: pip install ultralytics")
            return None, None
            
        except Exception as e:
            print(f"\n❌ 학습 중 오류 발생: {e}")
            return None, None
    
    def get_best_model_path(self):
        """학습된 최고 성능 모델 경로 반환"""
        best_model_path = self.config.RESULTS_DIR / self.config.RUN_NAME / 'weights' / 'best.pt'
        
        if best_model_path.exists():
            return best_model_path
        else:
            print(f"⚠ 최고 모델을 찾을 수 없습니다: {best_model_path}")
            return None