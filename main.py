# -*- coding: utf-8 -*-
"""
PCB 결함 탐지 - 메인 실행 파일
YOLOv11 기반 PCB 기판 결함 자동 탐지 시스템

Authors: 임주홍, 정명재
Date: 2024
"""

import sys
from pathlib import Path
import argparse


def print_header():
    """프로그램 헤더 출력"""
    print("\n" + "=" * 60)
    print(" " * 15 + "PCB 결함 탐지 시스템")
    print(" " * 12 + "YOLOv11 기반 자동화 검사")
    print("=" * 60)
    print("\n👥 Authors: 임주홍, 정명재")
    print("📅 Project: PCB Defect Detection using YOLO")
    print("=" * 60)

def print_section(title):
    """섹션 구분 출력"""
    print(f"\n{'#' * (len(title) + 6)}")
    print(f"## {title} ##")
    print(f"{'#' * (len(title) + 6)}")

def print_info(message):
    """정보 메시지 출력"""
    print(f"[INFO] {message}")

def print_warning(message):
    """경고 메시지 출력"""
    print(f"[WARN] {message}")

def print_error(message):
    """오류 메시지 출력"""
    print(f"[ERROR] {message}")

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

try:
    from config import Config
    from data import DataDownloader, DataPreprocessor, DatasetSplitter
    from models import ModelTrainer, ModelEvaluator, ModelInference
    from utils import Visualizer, FileUtils
except ImportError as e:
    print_error(f"필요한 모듈을 찾을 수 없습니다. 프로젝트 구조를 확인하세요: {e}")
    sys.exit(1)


def parse_arguments():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='PCB 결함 탐지 시스템 - YOLOv11 기반 자동화 검사'
    )
    
    parser.add_argument(
        '--skip-download', action='store_true', help='데이터 다운로드 단계 건너뛰기'
    )
    parser.add_argument(
        '--skip-preprocessing', action='store_true', help='데이터 전처리 단계 건너뛰기'
    )
    parser.add_argument(
        '--skip-training', action='store_true', help='모델 학습 단계 건너뛰기'
    )
    parser.add_argument(
        '--skip-visualization', action='store_true', help='시각화 단계 건너뛰기'
    )
    parser.add_argument(
        '--only-evaluate', action='store_true', help='평가만 수행 (기존 모델 사용)'
    )
    parser.add_argument(
        '--model-path', type=str, default=None, help='평가 및 추론에 사용할 모델 경로'
    )
    parser.add_argument(
        '--epochs', type=int, default=None, help='학습 에포크 수 (기본값: config.py 참조)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None, help='배치 크기 (기본값: config.py 참조)'
    )
    parser.add_argument(
        '--inference', action='store_true', help='테스트 이미지를 사용하여 추론(Inference) 단계 수행'
    )
    
    return parser.parse_args()


def step_1_setup():
    """단계 1: 환경 설정"""
    print_section("[단계 1] 환경 설정")
    
    Config.setup_directories()
    Config.print_config()
    
    print_info("환경 설정 완료")
    
    # 디스크 공간 확인
    workspace_size = FileUtils.get_directory_size(Config.WORKSPACE_ROOT)
    print_info(f"현재 작업 공간 크기: {workspace_size:.2f} MB")


def step_2_download(skip=False):
    """단계 2: 데이터 다운로드"""
    print_section("[단계 2] 데이터 다운로드")
    
    if skip:
        print_info("데이터 다운로드 단계를 건너뜁니다.")
        return True
    
    downloader = DataDownloader()
    
    # 다운로드 시도
    if not downloader.download():
        downloader.wait_for_manual_setup()
    
    # 데이터 검증
    if not downloader.verify_data():
        print_error("데이터 검증 실패")
        return False
    
    # 데이터 통계
    img_count = FileUtils.count_files(Config.IMAGES_DIR, '*.jpg')
    xml_count = FileUtils.count_files(Config.ANNOTATIONS_DIR, '*.xml')
    print_info(f"이미지 파일: {img_count}개")
    print_info(f"어노테이션 파일: {xml_count}개")
    
    return True


def step_3_preprocessing(skip=False):
    """단계 3: 데이터 전처리"""
    print_section("[단계 3] 데이터 전처리")
    
    if skip:
        print_info("데이터 전처리 단계를 건너뜁니다.")
        return None
    
    preprocessor = DataPreprocessor()
    
    # XML 파싱
    annotations_df = preprocessor.parse_xml_annotations()
    print_info(f"총 {len(annotations_df)}개의 객체 파싱 완료")
    
    # 클래스 분포 시각화
    try:
        Visualizer.plot_class_distribution(
            annotations_df,
            save_path=Config.PROJECT_DATA_ROOT / 'class_distribution.png'
        )
    except Exception as e:
        print_warning(f"클래스 분포 시각화 실패: {e}")
    
    # 이미지 리사이즈
    resized_annotations_df = preprocessor.resize_images()
    print_info("이미지 리사이즈 완료")
    
    return resized_annotations_df


def step_4_split(annotations_df):
    """단계 4: 데이터셋 분할"""
    print_section("[단계 4] 데이터셋 분할")
    
    # 전처리 건너뛰기 시, annotations_df가 None일 수 있으므로 이를 처리
    if annotations_df is None:
        print_error("전처리 단계에서 DataFrame을 가져오지 못했습니다. 파일을 확인하세요.")
        return None
        
    splitter = DatasetSplitter(annotations_df)
    
    # YOLO 형식 변환
    yolo_df = splitter.convert_to_yolo_format()
    print_info("YOLO 형식 변환 완료")
    
    # 데이터 분할
    splits = splitter.split_dataset()
    print_info(f"Train: {len(splits['train'])}장")
    print_info(f"Val: {len(splits['val'])}장")
    print_info(f"Test: {len(splits['test'])}장")
    
    # 분할 비율 시각화
    try:
        Visualizer.plot_split_distribution(
            splits,
            save_path=Config.PROJECT_DATA_ROOT / 'split_distribution.png'
        )
    except Exception as e:
        print_warning(f"분할 비율 시각화 실패: {e}")
    
    # 파일 저장
    splitter.save_split_data()
    yaml_path = splitter.create_yaml_file()
    print_info(f"데이터셋 저장 및 YAML 파일 생성 완료: {yaml_path}")
    
    return splits


def step_5_training(skip=False, epochs=None, batch_size=None):
    """단계 5: 모델 학습"""
    print_section("[단계 5] 모델 학습")
    
    if skip:
        print_info("모델 학습 단계를 건너뜁니다.")
        return None, None
    
    # 설정 오버라이드
    if epochs:
        Config.EPOCHS = epochs
        print_info(f"Epochs 변경: {epochs}")
    
    if batch_size:
        Config.BATCH_SIZE = batch_size
        print_info(f"Batch Size 변경: {batch_size}")
    
    trainer = ModelTrainer()
    model, results = trainer.train()
    
    if model is None:
        print_error("모델 학습 실패")
        return None, None
    
    print_info("모델 학습 완료")
    
    return trainer, model


def step_6_visualization(skip=False):
    """단계 6: 학습 결과 시각화"""
    print_section("[단계 6] 학습 결과 시각화")
    
    if skip:
        print_info("시각화 단계를 건너뜁니다.")
        return
    
    results_dir = Config.RESULTS_DIR / Config.RUN_NAME
    
    if not results_dir.exists():
        print_warning(f"결과 디렉토리를 찾을 수 없습니다: {results_dir}")
        return
    
    try:
        Visualizer.show_training_results(results_dir)
        print_info("학습 결과 시각화 완료")
    except Exception as e:
        print_error(f"시각화 중 오류 발생: {e}")


def get_best_model_path(trainer=None, model_path_arg=None):
    """최적 모델 경로를 결정하고 반환"""
    if model_path_arg:
        best_model_path = Path(model_path_arg)
        print_info(f"지정된 모델 경로 사용: {best_model_path}")
    elif trainer:
        best_model_path = trainer.get_best_model_path()
        print_info(f"학습된 최고 모델 경로 사용: {best_model_path}")
    else:
        # 기본 경로에서 찾기
        best_model_path = Config.RESULTS_DIR / Config.RUN_NAME / 'weights' / 'best.pt'
        print_info(f"기본 경로에서 최고 모델 찾기: {best_model_path}")

    if not best_model_path.exists():
        print_error(f"최고 모델 가중치 파일이 존재하지 않습니다: {best_model_path}")
        return None
        
    return best_model_path


def step_7_evaluation(trainer=None, model_path=None):
    """단계 7: Test 세트 최종 평가"""
    print_section("[단계 7] Test 세트 최종 평가")
    
    best_model_path = get_best_model_path(trainer, model_path)
    if best_model_path is None:
        return None

    # 평가 수행
    evaluator = ModelEvaluator(best_model_path)
    metrics = evaluator.evaluate()
    
    if metrics:
        # 주요 성능 지표 출력
        print_info("--- 최종 Test 세트 성능 ---")
        print_info(f"🚀 mAP@0.5 (느슨한 기준): {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print_info(f"🎯 mAP@0.5:0.95 (엄격한 기준): {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        print_info(f"✅ Precision: {metrics.get('metrics/precision(B)', 'N/A'):.4f}")
        print_info(f"🔍 Recall: {metrics.get('metrics/recall(B)', 'N/A'):.4f}")
        print_info("---------------------------")
    else:
        print_error("모델 평가 실패. 로그를 확인하세요.")
        
    return best_model_path


def step_8_inference(best_model_path):
    """단계 8: 추론 (실제 운영 시뮬레이션)"""
    print_section("[단계 8] 추론 (Inference)")

    if best_model_path is None:
        print_error("추론을 위한 모델 경로가 유효하지 않습니다.")
        return

    # 추론 수행
    inferencer = ModelInference(best_model_path)
    
    # Test 세트의 샘플 이미지들을 대상으로 추론
    input_dir = Config.PROCESSED_DATA_ROOT / 'images' / 'test'
    output_dir = Config.RESULTS_DIR / 'inference_output'
    
    # output_dir 생성
    output_dir.mkdir(parents=True, exist_ok=True)

    print_info(f"Test 세트 샘플 이미지에 대해 추론을 수행합니다.")
    print_info(f"결과 저장 위치: {output_dir}")
    
    inferencer.run_inference(input_dir, output_dir)
    print_info("추론 완료. 결과는 inference_output 폴더에서 확인하세요.")


def main():
    """메인 실행 함수"""
    args = parse_arguments()
    
    print_header()
    
    # 1. 환경 설정
    step_1_setup()
    
    # 2. 데이터 다운로드
    if not args.skip_download and not step_2_download(args.skip_download):
        return

    # 3. 데이터 전처리
    annotations_df = None
    if not args.only_evaluate:
        annotations_df = step_3_preprocessing(args.skip_preprocessing)
    
    # 4. 데이터셋 분할 (학습/평가에 필요한 YAML 파일이 없으면 실행)
    if not Config.YAML_PATH.exists() and not args.only_evaluate:
        splits = step_4_split(annotations_df)
        if splits is None:
             return
    
    trainer = None
    best_model_path = None
    
    # 5. 모델 학습
    if not args.skip_training and not args.only_evaluate:
        trainer, model = step_5_training(
            args.skip_training, 
            args.epochs, 
            args.batch_size
        )
    
    # 6. 학습 결과 시각화
    step_6_visualization(args.skip_visualization)
    
    # 7. Test 세트 최종 평가
    if args.only_evaluate or (trainer and not args.skip_training):
        best_model_path = step_7_evaluation(trainer, args.model_path)
        
    # 8. 추론 (Inference)
    if args.inference and best_model_path:
        step_8_inference(best_model_path)
        
    print_section("--- 프로그램 종료 ---")

if __name__ == '__main__':
    main()