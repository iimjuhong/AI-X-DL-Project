# -*- coding: utf-8 -*-
"""파일 및 디렉토리 관리 유틸리티"""

import shutil
from pathlib import Path
import json
import yaml


class FileUtils:
    """파일 시스템 관련 유틸리티 클래스"""
    
    @staticmethod
    def clean_directory(directory):
        """
        디렉토리를 삭제하고 다시 생성
        
        Args:
            directory: 디렉토리 경로
        """
        directory = Path(directory)
        if directory.exists():
            print(f"  → 기존 디렉토리 삭제: {directory}")
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  → 디렉토리 생성: {directory}")
    
    @staticmethod
    def ensure_directory(directory):
        """
        디렉토리가 존재하지 않으면 생성
        
        Args:
            directory: 디렉토리 경로
        """
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def count_files(directory, pattern='*'):
        """
        디렉토리 내 특정 패턴의 파일 개수 세기
        
        Args:
            directory: 디렉토리 경로
            pattern: 파일 패턴 (예: '*.jpg', '*.xml')
            
        Returns:
            int: 파일 개수
        """
        directory = Path(directory)
        if not directory.exists():
            return 0
        return len(list(directory.glob(pattern)))
    
    @staticmethod
    def copy_file(src, dst):
        """
        파일 복사
        
        Args:
            src: 원본 파일 경로
            dst: 대상 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            print(f"  ⚠ 원본 파일이 없습니다: {src}")
            return False
        
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            return True
        except Exception as e:
            print(f"  ❌ 파일 복사 실패: {e}")
            return False
    
    @staticmethod
    def list_files(directory, pattern='*', recursive=False):
        """
        디렉토리 내 파일 목록 반환
        
        Args:
            directory: 디렉토리 경로
            pattern: 파일 패턴
            recursive: 하위 디렉토리 포함 여부
            
        Returns:
            list: 파일 경로 리스트
        """
        directory = Path(directory)
        if not directory.exists():
            return []
        
        if recursive:
            return list(directory.rglob(pattern))
        else:
            return list(directory.glob(pattern))
    
    @staticmethod
    def get_directory_size(directory):
        """
        디렉토리 전체 크기 계산 (MB)
        
        Args:
            directory: 디렉토리 경로
            
        Returns:
            float: 크기 (MB)
        """
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)  # MB로 변환
    
    @staticmethod
    def save_json(data, filepath):
        """
        JSON 파일로 저장
        
        Args:
            data: 저장할 데이터
            filepath: 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ JSON 저장 완료: {filepath}")
    
    @staticmethod
    def load_json(filepath):
        """
        JSON 파일 로드
        
        Args:
            filepath: 파일 경로
            
        Returns:
            dict: 로드된 데이터
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"  ⚠ JSON 파일이 없습니다: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_yaml(data, filepath):
        """
        YAML 파일로 저장
        
        Args:
            data: 저장할 데이터
            filepath: 파일 경로
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"  ✓ YAML 저장 완료: {filepath}")
    
    @staticmethod
    def load_yaml(filepath):
        """
        YAML 파일 로드
        
        Args:
            filepath: 파일 경로
            
        Returns:
            dict: 로드된 데이터
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"  ⚠ YAML 파일이 없습니다: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)