# -*- coding: utf-8 -*-
"""데이터 다운로드 모듈"""

import shutil
import kagglehub
from pathlib import Path
from config import Config


class DataDownloader:
    """Kaggle 데이터셋 다운로드"""
    
    def __init__(self):
        self.config = Config
    
    def download(self):
        """데이터셋 다운로드 및 준비"""
        print("\n" + "=" * 60)
        print("데이터셋 다운로드")
        print("=" * 60)
        
        # Kaggle API 시도
        try:
            import kagglehub
            print("✓ kagglehub 발견. 자동 다운로드를 시도합니다...")
            kaggle_path = Path(kagglehub.dataset_download(self.config.KAGGLE_DATASET_NAME))
            
            # 데이터 복사
            shutil.copytree(
                kaggle_path / 'PCB_DATASET',
                self.config.RAW_DATA_PATH,
                dirs_exist_ok=True
            )
            print("✓ 자동 다운로드 완료!")
            return True
            
        except ImportError:
            print("\n kagglehub가 설치되지 않았습니다.")
            self._print_manual_instructions()
            return False
        except Exception as e:
            print(f"\n⚠ 자동 다운로드 실패: {e}")
            self._print_manual_instructions()
            return False
    
    def _print_manual_instructions(self):
        """수동 다운로드 안내"""
        print("\n수동 다운로드 방법:")
        print("=" * 60)
        print("방법 1: Kaggle API 사용")
        print("  1) pip install kaggle")
        print("  2) Kaggle API 토큰 설정 (kaggle.json)")
        print(f"  3) kaggle datasets download -d {self.config.KAGGLE_DATASET_NAME}")
        print("\n방법 2: 수동 다운로드")
        print(f"  1) https://www.kaggle.com/datasets/{self.config.KAGGLE_DATASET_NAME}")
        print("  2) 다운로드 후 아래 경로에 압축 해제:")
        print(f"     {self.config.RAW_DATA_PATH}")
        print("=" * 60)
    
    def verify_data(self):
        """데이터 존재 확인"""
        if not self.config.IMAGES_DIR.exists():
            print(f"❌ 이미지 폴더를 찾을 수 없습니다: {self.config.IMAGES_DIR}")
            return False
        
        if not self.config.ANNOTATIONS_DIR.exists():
            print(f"❌ 어노테이션 폴더를 찾을 수 없습니다: {self.config.ANNOTATIONS_DIR}")
            return False
        
        print("✓ 데이터 검증 완료")
        return True
    
    def wait_for_manual_setup(self):
        """수동 설정 대기"""
        if not self.config.RAW_DATA_PATH.exists():
            input(f"\n'{self.config.RAW_DATA_PATH}' 경로에 데이터를 준비한 후 Enter를 눌러주세요...")