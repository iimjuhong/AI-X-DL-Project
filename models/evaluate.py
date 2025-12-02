# -*- coding: utf-8 -*-
"""모델 평가 모듈"""

from pathlib import Path
from config import Config


class ModelEvaluator:
    """YOLO 모델 평가"""
    
    def __init__(self, model_path=None):
        self.config = Config
        self.model_path = model_path
        self.model = None
        self.metrics = None
    
    def load_model(self, model_path=None):
        """모델 로드"""
        if model_path:
            self.model_path = model_path
        
        if not self.model_path or not Path(self.model_path).exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {self.model_path}")
            return False
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            print(f"✓ 모델 로드 완료: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def evaluate_test_set(self):
        """Test 세트 평가"""
        print("\n" + "=" * 60)
        print("Test 세트 최종 평가")
        print("=" * 60)
        
        if not self.model:
            print("❌ 모델이 로드되지 않았습니다.")
            return None
        
        try:
            print("Test 세트 평가 중...")
            self.metrics = self.model.val(
                data=str(self.config.YAML_PATH),
                split='test',
                plots=True
            )
            
            self._print_metrics()
            
            return self.metrics
            
        except Exception as e:
            print(f"❌ 평가 중 오류 발생: {e}")
            return None
    
    def _print_metrics(self):
        """평가 결과 출력"""
        if not self.metrics:
            return
        
        print("\n" + "=" * 60)
        print("최종 Test Set 성능")
        print("=" * 60)
        print(f"mAP@0.5:      {self.metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {self.metrics.box.map:.4f}")
        print(f"Precision:    {self.metrics.box.mp:.4f}")
        print(f"Recall:       {self.metrics.box.mr:.4f}")
        print("=" * 60)
    
    def get_results_dir(self):
        """결과 저장 디렉토리 반환"""
        if self.metrics:
            return Path(self.metrics.save_dir)
        return None