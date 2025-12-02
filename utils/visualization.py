# -*- coding: utf-8 -*-
"""시각화 도구"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import pandas as pd


class Visualizer:
    """학습 결과 및 데이터 시각화 클래스"""
    
    @staticmethod
    def show_image(image_path, title="", figsize=(12, 10), save_path=None):
        """
        이미지 표시
        
        Args:
            image_path: 이미지 파일 경로
            title: 제목
            figsize: 그림 크기
            save_path: 저장 경로 (선택사항)
            
        Returns:
            bool: 성공 여부
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"  ⚠ 이미지를 찾을 수 없음: {image_path.name}")
            return False
        
        try:
            img = mpimg.imread(str(image_path))
            
            plt.figure(figsize=figsize)
            plt.imshow(img)
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  → 저장됨: {save_path}")
            
            plt.show()
            print(f"  ✓ 표시 완료: {title}")
            return True
            
        except Exception as e:
            print(f"  ❌ 이미지 표시 실패: {e}")
            return False
    
    @staticmethod
    def show_training_results(results_dir):
        """
        학습 결과 이미지들을 순차적으로 표시
        
        Args:
            results_dir: 결과가 저장된 디렉토리
        """
        results_dir = Path(results_dir)
        
        if not results_dir.exists():
            print(f"\n  ⚠ 결과 디렉토리를 찾을 수 없습니다: {results_dir}")
            return
        
        print("\n" + "=" * 60)
        print("학습 결과 시각화")
        print("=" * 60)
        
        # 표시할 이미지 목록 (파일명, 제목)
        images_to_show = [
            ('results.png', 'Training Results Summary (Loss & Metrics)'),
            ('confusion_matrix_normalized.png', 'Normalized Confusion Matrix (Validation)'),
            ('confusion_matrix.png', 'Confusion Matrix (Validation)'),
            ('PR_curve.png', 'Precision-Recall Curve'),
            ('P_curve.png', 'Precision Curve'),
            ('R_curve.png', 'Recall Curve'),
            ('F1_curve.png', 'F1 Score Curve'),
        ]
        
        for img_name, title in images_to_show:
            img_path = results_dir / img_name
            Visualizer.show_image(img_path, title)
        
        # 검증 예측 샘플 이미지
        print("\n검증 예측 샘플:")
        pred_images = sorted(list(results_dir.glob('val_batch*_pred.jpg')))
        
        if pred_images:
            print(f"  총 {len(pred_images)}개의 검증 샘플 발견 (최대 3개 표시)")
            for i, img_path in enumerate(pred_images[:3]):
                Visualizer.show_image(
                    img_path,
                    f'Validation Prediction Sample #{i+1}',
                    figsize=(15, 15)
                )
        else:
            print("  ⚠ 검증 예측 이미지를 찾을 수 없습니다.")
    
    @staticmethod
    def show_test_results(results_dir):
        """
        Test 세트 결과 이미지들을 표시
        
        Args:
            results_dir: Test 결과가 저장된 디렉토리
        """
        results_dir = Path(results_dir)
        
        if not results_dir.exists():
            print(f"\n  ⚠ Test 결과 디렉토리를 찾을 수 없습니다: {results_dir}")
            return
        
        print("\n" + "=" * 60)
        print("Test 세트 결과 시각화")
        print("=" * 60)
        
        images_to_show = [
            ('confusion_matrix_normalized.png', 'Test Set - Normalized Confusion Matrix'),
            ('confusion_matrix.png', 'Test Set - Confusion Matrix'),
            ('PR_curve.png', 'Test Set - Precision-Recall Curve'),
            ('P_curve.png', 'Test Set - Precision Curve'),
            ('R_curve.png', 'Test Set - Recall Curve'),
            ('F1_curve.png', 'Test Set - F1 Score Curve'),
        ]
        
        for img_name, title in images_to_show:
            img_path = results_dir / img_name
            Visualizer.show_image(img_path, title)
    
    @staticmethod
    def plot_class_distribution(annotations_df, save_path=None):
        """
        클래스별 데이터 분포 시각화
        
        Args:
            annotations_df: 어노테이션 DataFrame
            save_path: 저장 경로 (선택사항)
        """
        plt.figure(figsize=(12, 6))
        
        class_counts = annotations_df['class'].value_counts()
        
        plt.bar(class_counts.index, class_counts.values, color='steelblue', edgecolor='black')
        plt.xlabel('Defect Class', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('PCB Defect Class Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # 막대 위에 개수 표시
        for i, (class_name, count) in enumerate(class_counts.items()):
            plt.text(i, count + 5, str(count), ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  → 저장됨: {save_path}")
        
        plt.show()
        print("  ✓ 클래스 분포 시각화 완료")
    
    @staticmethod
    def plot_split_distribution(splits, save_path=None):
        """
        Train/Val/Test 분할 비율 시각화
        
        Args:
            splits: {'train': [...], 'val': [...], 'test': [...]}
            save_path: 저장 경로 (선택사항)
        """
        split_counts = {k: len(v) for k, v in splits.items()}
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 막대 그래프
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        ax1.bar(split_counts.keys(), split_counts.values(), color=colors, edgecolor='black')
        ax1.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax1.set_title('Dataset Split Distribution (Bar)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for i, (split, count) in enumerate(split_counts.items()):
            ax1.text(i, count + 10, str(count), ha='center', fontweight='bold')
        
        # 파이 차트
        ax2.pie(split_counts.values(), labels=split_counts.keys(), autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontweight': 'bold'})
        ax2.set_title('Dataset Split Distribution (Pie)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  → 저장됨: {save_path}")
        
        plt.show()
        print("  ✓ 데이터 분할 시각화 완료")
    
    @staticmethod
    def show_sample_images(image_dir, num_samples=6, figsize=(15, 10)):
        """
        샘플 이미지들을 그리드로 표시
        
        Args:
            image_dir: 이미지 디렉토리
            num_samples: 표시할 샘플 수
            figsize: 그림 크기
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg'))[:num_samples]
        
        if not image_files:
            print(f"  ⚠ 이미지를 찾을 수 없습니다: {image_dir}")
            return
        
        rows = (num_samples + 2) // 3
        cols = min(3, num_samples)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)
        
        for idx, img_path in enumerate(image_files):
            row = idx // cols
            col = idx % cols
            
            img = mpimg.imread(str(img_path))
            axes[row, col].imshow(img)
            axes[row, col].set_title(img_path.name, fontsize=10)
            axes[row, col].axis('off')
        
        # 빈 subplot 숨기기
        for idx in range(len(image_files), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Sample Images', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        print(f"  ✓ {len(image_files)}개의 샘플 이미지 표시 완료")