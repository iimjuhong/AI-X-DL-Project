# YOLO 기반 PCB기판 결함 탐색
### Members
- 임주홍 전기공학 juehong9253@gmail.com
- 정명재 융합전자공학 qwertymike@hanyang.ac.kr

## 📌 1. Proposal (Option1)
### 1.1 배경 및 동기
PCB(인쇄회로기판)은 현대 전자제품 제조 공정에 많이 쓰이는 회로입니다. 산업 현장에서는 머신러닝 패턴 인식을 활용한 자동 광학 검사를 주로 사용해 왔습니다. 하지만, 기존 머신러닝 기반 자동광학검사(AOI)는 spur와 같은 비정형 결함 탐지에 명확한 한계를 가지고 있습니다. 특히 5G 통신 및 AI 연산 기능이 고도화됨에 따라 회로 패턴이 초미세화되면서, 기존 방식으로는 탐지하기 어려운 미세 결함이 증가하고 있습니다. 이러한 배경에서, 딥러닝 기반의 객체 탐지 기술은 기존 검사 방식의 유연성 부족과 비일관성 문제를 해결할 강력한 대안으로 산업 현장에서 연구 중입니다.

***

### 1.2 목적
본 프로젝트는 YOLO 아키텍처를 기반으로, 6가지 특정 PCB 결함을 신속하고 정밀하게 구분해내는 자동화 시스템 구축을 목표로 합니다.
YOLO 모델의 학습 및 구현을 넘어, 실제 산업 현장에서의 적용 신뢰도를 보증하는 것을 중점을 두었습니다. 최종 목적은 결함의 유무 판정을 넘어, 판정 부분 비교를 위한 정량적 데이터(mAP, Precision, Recall)를 제공함으로써 공정 개선에 기여하는 것입니다.


## 📌 2. Datasets
### 2.1 데이터 수집
본 프로젝트는 Kaggle의 'PCB Defects' 데이터셋(akhatova/pcb-defects)을 사용하였습니다. 

이 데이터셋은 6가지 주요 PCB 결함 유형을 포함하고 있습니다. 각 클래스는 제조 공정상의 각기 다른 문제를 대변합니다.
- missing_hole
- mouse_bite
- open_circuit
- short
- spur
- spurious_copper

<img width="1920" height="1920" alt="image" src="https://github.com/user-attachments/assets/f3a4ca0d-3739-4a20-b253-78a2b137826b" />

이 중 spur(회로에서 뾰족하게 튀어나온 돌기)와 spurious_copper(회로와 무관하게 잔류하는 구리 조각)는 모두 '불필요한 구리'라는 시각적 유사성을 공유하며, 형태가 비정형적이고 크기가 미세하여 모델이 탐지하고 분류하기에 가장 까다로운 유형으로 분석되었습니다.

***

### 2.2 데이터 전처리
- XML 파싱: 수백 개의 개별 어노테이션 파일(.xml)을 파싱하여, 파일명, 원본 이미지 크기(width, height), 결함 클래스, 바운딩 박스 좌표(xmin, ymin, xmax, ymax) 정보를 추출하였습니다. 이 모든 정보를 단일 Pandas DataFrame으로 변환하여 데이터 관리를 용이하게 하였습니다.
- 이미지 리사이즈: YOLOv11 모델의 표준 입력 크기이자, 연산 효율성과 탐지 성능 간의 균형을 고려하여 모든 이미지를 640x640 픽셀로 일괄 리사이즈하였습니다.
- 좌표 스케일링: 원본 이미지 크기에 맞춰져 있던 바운딩 박스 좌표를, 2단계에서 리사이즈된 640x640 크기에 맞게 비례적으로 스케일링하였습니다.
- YOLO 포맷 변환: YOLO 학습에 필요한 텍스트(.txt) 라벨 형식(class_id, x_center, y_center, width, height - 모든 값은 0과 1 사이로 정규화됨)으로 변환하였습니다.
- 데이터 분할: 전체 데이터셋을 학습(Train), 검증(Validation), 테스트(Test) 셋으로 8:1:1의 비율로 명확히 분리하여 저장하였습니다. 테스트 세트는 모델의 최종 성능 평가를 위해서만 사용되며, 학습 및 검증 과정에는 일절 관여하지 않도록 하여 평가의 객관성을 확보하였습니다.

***

### 2.3 데이터 증강
모델의 일반화 성능과 강인성을 향상시키기 위해 다양한 데이터 증강 기법을 적용하였습니다. 
- Mixup(두 이미지 혼합) 및 HSV(색상, 채도, 명도) 변환 : 실제 현장에서 발생할 수 있는 다양한 조명 변화나 카메라 노이즈 환경에 대응할 수 있게 합니다.
- Flip(좌우 반전) 및 Degrees(회전) : PCB가 검사 라인에 놓이는 각도가 미세하게 비틀어지더라도 모델이 일관된 성능을 내도록 유도하였습니다.

이러한 증강 기법은 한정된 데이터셋으로도 수만 장의 데이터를 학습하는 것과 유사한 효과를 내어 과적합을 방지합니다.


## 📌 3. Methodology
PCB 결함은 제품의 신뢰도와 직결되므로 신속하고 정확한 검출이 필수적입니다. 따라서, 실시간성과 높은 정확도를 강점으로 하는 YOLO(You Only Look Once) 아키텍처를 핵심 방법론으로 튜닝하는 방식을 채택하였습니다.
### 3.1 데이터셋 분할
전체 데이터셋은 총 1386개의 이미지와 6가지의 결함 라벨을 포함합니다. 모델의 학습, 검증, 평가를 위해 데이터셋은 임의로 각각 8:1:1 비율로 분리하였습니다.
- 학습(Train) 세트: 1108개 이미지 
- 검증(Validation) 세트: 138개 이미지
- 테스트(Test) 세트: 138개 이미지
분할 구조를 통해 모델이 충분한 양의 데이터로 학습하고, 학습 과정에서 과적합(overfitting)을 모니터링하며, 최종적으로 학습에 사용되지 않은 데이터를 통해 일반화 성능을 객관적으로 평가할 수 있도록 설계하였습니다.

```python
unique_filenames = yolo_df['filename'].unique()
np.random.shuffle(unique_filenames)

train_end = int(len(unique_filenames) * 0.8)
val_end = train_end + int(len(unique_filenames) * 0.1)

splits = {
    'train': unique_filenames[:train_end],
    'val': unique_filenames[train_end:val_end],
    'test': unique_filenames[val_end:]
}

print("파일 이동 및 라벨 파일 생성 중")
for split, filenames in splits.items():
    for fname in tqdm(filenames, desc=f"Processing {split}"):
        # 이미지 이동 (Copy)
        src_img = resized_img_dir / fname
        dst_img = output_dir_processed / 'images' / split / fname
        if src_img.exists():
            shutil.copy(src_img, dst_img)

        # 라벨 생성
        file_objects = yolo_df[yolo_df['filename'] == fname]
        label_path = output_dir_processed / 'labels' / split / f"{Path(fname).stem}.txt"

        with open(label_path, 'w') as f:
            for _, row in file_objects.iterrows():
                f.write(f"{int(row['class_id'])} {row['x_center']:.6f} {row['y_center']:.6f} {row['bbox_w']:.6f} {row['bbox_h']:.6f}\n")
```


***

### 3.2 모델 아키텍처 정의
본 프로젝트는 ultralytics 라이브러리에서 제공하는 YOLO v11 모델 아키텍처의 'small' (s) 버전을 기반으로 한 커스텀 모델로, 속도와 정확도 간의 균형이 뛰어나 실제 산업 현장의 실시간 검사 시스템에 적용하기에 적합하다고 판단하여 선정하였습니다.
- YOLO v11s는 경량화된(small) 버전으로, 백본(Backbone)과 넥(Neck), 헤드(Head)로 구성됩니다.
- Backbone(특징 추출 역할) : 입력 이미지를 받아 다양한 수준의 특징 맵(feature map)을 만듭니다. 저수준(모서리, 질감) 특징부터 고수준(형태, 의미) 특징까지 추출합니다.
- Neck(특징 융합) : Backbone에서 나온 서로 다른 크기의 특징 맵들을 결합하여 객체 탐지에 더 유용한 특징을 만듭니다.
- Head(검출) : Neck에서 전달받은 3가지 스케일의 특징 맵을 입력받아 실제 객체 탐지를 수행
- Backbone에서 특징 추출하고 Neck에서 이 특징들을 효율적으로 섞어준 뒤, Head에서 3가지 다른 크기의 특징 맵을 보고 다양한 크기의 객체를 최종적으로 탐지하는 구조입니다.

<img width="1581" height="1505" alt="image" src="https://github.com/user-attachments/assets/69e7166b-3b36-4b2b-b9a4-3f841a5b0015" />

***

### 3.3 하이퍼파라미터
- batch_size: 16 
- learning_rate: 0.001
- epochs: 100
- optimizer: Adam
- image_size: 640
- hsv_h : 0.015
- hsv_s : 0.7
- hsv_v : 0.4
- degrees : 10
- mixup : 0.3

***

### 3.4 학습 프로세스
이 학습 과정은 사전 훈련된(pre-trained) 가중치를 가진 yolo11s.pt 모델, 즉 YOLOv8-S (Small) 모델의 경량화 버전을 기반으로 시작하는 전이 학습(Transfer Learning) 전략을 채택했습니다. 이는 대규모 데이터셋(ImageNet, COCO)을 통해 이미 일반적인 시각적 특징을 학습한 모델을 가져와, 적은 양의 PCB 결함 데이터만으로도 해당 특정 도메인에 빠르게 적응시키고 높은 탐지 성능을 확보하기 위함입니다.

```python
results_base_dir_colab = Path('/content/pcb_results')
dest_results_dir_drive = project_root / 'results'

model = YOLO('yolo11s.pt')

try:
    results = model.train(
        data=str(yaml_path),
        epochs=100,
        batch=16,
        imgsz=640,
        project=str(results_base_dir_colab),
        name='yolo11s_run',
        optimizer='Adam'
        exist_ok=True,
        hsv_h=0.015  # Hue (색조): -0.015~0.015 범위로 변화
        hsv_s=0.7    # Saturation (채도): -0.7~0.7 범위로 변화  
        hsv_v=0.4    # Value (명도): -0.4~0.4 범위로 변화
        degrees=10.0,
        fliplr=0.0,
        mixup=0.3
    )

    # 결과 복사
    print("학습 결과를 Google Drive로 복사 중")
    source_dir = results_base_dir_colab / 'yolo11s_run'
    dest_dir = dest_results_dir_drive / 'yolo11s_run'
    shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
    print(f"모든 과정 완료! 결과 위치: {dest_dir}")

except Exception as e:
    print(f"\n학습 중 오류 발생: {e}")
```
***

### 3.5 모델 성능 평가 (검증)
학습 과정에서 validation 세트를 사용하여 성능을 측정하고 최적 모델을 선정하는 절차는 전이 학습의 효과를 극대화하고, 모델의 일반화 능력을 보장하는 데 필수적입니다. 각 에포크가 끝날 때마다 모델은 가중치 업데이트를 멈추고 val 데이터셋을 사용하여 현재 가중치의 성능을 측정합니다. model.train() 함수 호출을 통해 100 에포크 동안 data.yaml의 val 경로에 지정된 데이터셋을 사용하여 자동으로 훈련과 검증을 반복하도록 설정하였습니다. 이 검증 과정은 다음과 같은 역할을 수행합니다.
- 과적합 방지: 모델이 훈련 데이터에만 너무 잘 맞추어지는 현상(과적합)이 발생하는지 실시간으로 확인합니다.
- 최적 모델 선정: 측정된 validation 성능이 이전 에포크보다 높을 경우, 해당 시점의 모델 가중치(weights)를 best.pt 파일로 저장합니다.
- 학습 종료: 100 에포크가 모두 끝난 후, 최종적으로 best.pt는 전체 학습 과정 중 validation 세트에서 가장 좋은 성능을 낸 모델이 됩니다.

***

### 3.6 최종 모델 성능 평가 (테스트)
로드된 모델의 val() 메소드를 test 데이터셋에 대해 실행하였습니다. 이 과정에서 모델은 test 이미지를 입력받아 결함을 예측하고, 이 예측 결과(바운딩 박스, 클래스, 신뢰도 점수)를 실제 정답 라벨과 비교합니다. 평가가 완료되면, 객체 탐지 모델의 표준 성능 지표인 mAP가 산출됩니다. 구체적으로, IoU(Intersection over Union) 임계값에 따른 mAP@0.5 (느슨한 기준) 및 mAP@0.5:0.95 (엄격한 기준) 값을 통해 모델의 정확도를 정량적으로 평가합니다. 이와 더불어 정밀도(Precision)와 재현율(Recall)을 확인하여, 모델이 결함을 놓치지 않고(높은 재현율) 정확하게 예측하는지(높은 정밀도)를 종합적으로 분석하였습니다. 또한 모델의 오분류 경향을 클래스별로 한눈에 보여주는 Confusion Matrix까지 불러옵니다.

```python
metrics = model.val(data=str(yaml_file_path), split='test', plots=True)
save_dir = Path(metrics.save_dir)

def show_result_image(img_path, title):
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.title(title, fontsize=15)
            plt.axis('off')
            plt.show()
        else:
            print(f" 이미지를 찾을 수 없습니다: {img_path}")
```

***

## 📌 4. Evaluation & Analysis
### 4.1 Train loss / Validation loss / Precision & Recall / mAP
- 학습 손실(Loss) 및 수렴 안정성 분석:
학습이 진행된 100 Epoch 동안의 손실 그래프를 분석한 결과, 모델은 매우 안정적인 학습 양상을 보였습니다. 
Train Loss: box_loss(객체 위치), cls_loss(객체 분류), dfl_loss(분포 초점)의 세 가지 지표 모두 학습 초기(0~10 Epoch)에 급격하게 감소한 후, 이후 완만한 하강 곡선을 그리며 특정 값에 수렴했습니다. 
이는 모델이 PCB 결함 데이터의 특징을 빠르게 학습했음을 의미합니다. 

- Validation Loss: 검증 데이터셋에 대한 손실값(val/box_loss, val/cls_loss, val/dfl_loss) 역시 Train Loss와 동기화되어 지속적으로 감소했습니다. 특히 학습 후반부(80~100 Epoch)에서도 검증 손실이 다시 증가(튀는 현상)하지 않고 낮게 유지되고 있습니다. 
이는 딥러닝 모델 학습 시 가장 경계해야 할 '과적합(Overfitting)' 현상이 발생하지 않았음을 강력하게 시사합니다. 즉, 본 모델은 학습 데이터에만 편향되지 않고 새로운 PCB 이미지에 대해서도 일반화된 탐지 성능을 기대할 수 있습니다.

- 정밀도(Precision) 및 재현율(Recall) 분석:
Metrics 그래프와 Precision-Recall Curve를 통해 모델의 분류 및 탐지 성능을 구체적으로 확인할 수 있습니다. 
전반적 추세: metrics/precision(B)과 metrics/recall(B)은 학습 초기에 빠르게 0.8 이상으로 진입했으며, 최종적으로 두 지표 모두 0.9(90%)를 상회하는 높은 구간에서 안정화되었습니다.
클래스별 성능 (PR Curve 해석): 전체 클래스(all classes)에 대한 mAP@0.5는 0.967로 매우 우수한 성능을 기록했습니다. 
최상위 성능: missing_hole(0.995), open_circuit(0.991), mouse_bite(0.985)와 같은 결함들은 높은 정확도로 탐지하고 있습니다. 이는 해당 결함들의 시각적 특징이 뚜렷하여 모델이 이를 명확히 구분하고 있음을 보여줍니다. 
상대적 취약 클래스: spur(0.912)와 spurious_copper(0.942)는 다른 클래스에 비해 상대적으로 낮은 점수를 기록했습니다. 이는 상대적으로 미세한 구리 잔여물이나 돌기(spur)의 형태가 배경이나 다른 패턴과 유사하여 탐지 난이도가 높기 때문으로 해석됩니다. 하지만 이 역시 0.9 이상의 높은 수치이므로 실용적인 탐지에는 문제가 없는 수준입니다.

- mAP(mean Average Precision) 및 위치 정확도:
mAP50(B): IoU(Intersection over Union) 임계값을 0.5로 설정했을 때의 성능인 metrics/mAP50(B)는 최종적으로 약 0.97에 도달했습니다. 이는 예측한 바운딩 박스와 실제 정답 박스가 50% 이상 겹치는 정답을 찾아내는 능력이 탁월함을 의미합니다. mAP50-95(B): IoU 임계값을 0.5부터 0.95까지 엄격하게 적용하여 평균을 낸 metrics/mAP50-95(B)는 약 0.53~0.55 수준으로 수렴했습니다.

- F1 Score 및 최적 임계값(Threshold) 선정:
F1 Score는 정밀도와 재현율의 조화 평균으로, 모델의 균형 잡힌 성능을 보여줍니다. 제공된 F1-Confidence Curve에 따르면, 모든 클래스에 대한 최고 F1 Score는 0.94이며, 이때의 Confidence Threshold(신뢰도 임계값)는 0.311입니다. 이는 실무 적용 시 모델의 신뢰도(Confidence) 기준을 0.311로 설정했을 때, 오탐지(False Positive)와 미탐지(False Negative) 사이에서 가장 이상적인 성능 균형을 맞추고 있습니다.

<img width="1789" height="590" alt="image" src="https://github.com/user-attachments/assets/36efb3f3-d8af-48c0-a9c1-27ec78de6b2f" />
<img width="950" height="509" alt="image" src="https://github.com/user-attachments/assets/a625413f-0c12-43ee-bd59-7ed53afb098b" />
<img width="950" height="664" alt="image" src="https://github.com/user-attachments/assets/252b1ae4-bbd7-4ea6-8d05-4d64ec126f43" />
<img width="950" height="664" alt="image" src="https://github.com/user-attachments/assets/e226d509-19d6-416f-ae3b-96b391a8a785" />
<img width="1175" height="1199" alt="image" src="https://github.com/user-attachments/assets/ea84b578-3011-4e0b-9350-65ff27ad7d83" />
<img width="1175" height="1199" alt="image" src="https://github.com/user-attachments/assets/68cfa1b6-9332-4c14-bd23-537f025163cb" />
<img width="1175" height="1199" alt="image" src="https://github.com/user-attachments/assets/f16b8011-8121-4af8-84f5-3462fba00dc3" />

***

### 4.2 Normalized Confusion Matrix
- missing_hole : 100%
- spurious_copper : 96%
- short : 93%
- spur : 85%
- mouse_bite / open_circuit : 81%
- background : 0%
클래스별 예측 성능 혼동 행렬 분석 결과,
missing_hole은 1.00(100%)의 완벽한 분류 성능을 보였으며, open_circuit과 short는 0.96, mouse_bite와 spurious_copper는 0.94로 대부분의 클래스에서 매우 높은 정답률을 기록했습니다.
다만 spur의 경우 0.82로, 실제 결함 중 약 18%를 background으로 잘못 예측하는 False Negative가 발생하여 상대적으로 가장 낮은 성능을 보였습니다.
Background 클래스에서 가장 주목할 점은 정상 클래스의 예측 성공률이 0% 라는 것입니다. 행렬 우측 열을 보면, 모델이 정상 배경을 mouse_bite(0.25), spur(0.25) 등의 결함으로 잘못 예측하는 False Positive 현상이 뚜렷합니다. 이는 바운딩 박스(Bounding Box) 어노테이션의 구조적 한계에 기인한 것으로 판단됩니다. 직사각형 박스는 결함뿐만 아니라 주변의 정상 회로 픽셀까지 포함하게 됩니다. 이로 인해 모델이 박스 내부의 정상 패턴까지 결함의 특징으로 학습하게 되어, 실제 정상 이미지에서 유사한 복잡한 패턴이 나타날 때 이를 결함으로 오인하는 오류를 범하고 있습니다.
<img width="950" height="741" alt="image" src="https://github.com/user-attachments/assets/9a0a1cf0-e78f-408b-88a2-38df259992b7" />
*** 

### 4.3 Test set 최종 평가
정량 평가: (PR & F1 Curve) 학습에 관여하지 않은 테스트 셋 평가 결과, 모델은 mAP@0.5 기준 0.950을 기록하며 우수한 일반화 성능을 입증했습니다.
F1-Confidence Curve 분석 시, 신뢰도 임계값(Confidence Threshold) 0.353에서 최대 F1 점수 0.95를 달성하여 최적의 동작 지점을 확인했습니다. 
클래스별로는 open_circuit(0.984)과 missing_hole(0.961)이 가장 높은 성능을 보인 반면, spur(0.911)는 상대적으로 가장 낮은 mAP를 기록했습니다. spur의 PR 곡선이 다른 클래스보다 안쪽으로 쳐지는 형태는 해당 결함 탐지의 난이도가 높음을 시각적으로 보여줍니다. 
정성 평가: (Confusion Matrix 연계) 성능이 가장 낮은 spur 클래스를 심층 분석한 결과, 
short나 spurious_copper와의 혼동보다는 Background로 잘못 예측(18%)하는 비율이 압도적으로 높았습니다. 이는 spur 결함이 미세하고 비정형적이라 모델이 이를 놓치고 지나가는 경우 (False Negative)가 빈번함을 시사합니다. 따라서 spur와 spurious_copper 간의 유사성보다는, 미세한 spur 객체와 배경을 구분하는 식별력을 강화하는 방향으로 개선이 필요합니다.
<img width="950" height="664" alt="image" src="https://github.com/user-attachments/assets/c28354c0-ffb9-4c97-9e77-b760f9689249" />
<img width="950" height="664" alt="image" src="https://github.com/user-attachments/assets/7b7b33f5-0d7b-4d41-8ae5-046685e9ab9e" />
<img width="950" height="741" alt="image" src="https://github.com/user-attachments/assets/36a632d0-c11d-42db-a284-b954f034b641" />


## 📌 5. Related Works
* J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?," in Advances in Neural Information Processing Systems (NIPS), 2014.
* Ultralytics, "YOLOv11 Documentation.
* J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in CVPR, 2016.
* I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization (AdamW)," in ICLR, 2019.

## 📌 6. Conclusion
### 6.1 요약 및 성과
본 프로젝트에서는 Full Fine-tuning, AdamW 옵티마이저 선정과 같은 체계적인 딥러닝 파이프라인을 통해 YOLOv11 기반의 PCB 결함 탐지 모델을 성공적으로 구현하였습니다. 테스트 셋 평가 결과, mAP@0.5 0.973, 정밀도 0.967, 재현율 0.939라는 우수한 성과를 달성했습니다. 이는 본 모델이 육안 검사를 대체하여 결함을 놓치지 않고(높은 재현율) 정확하게(높은 정밀도) 찾아내는 자동화 시스템으로서의 높은 잠재력을 가지고 있음을 입증합니다. 본 프로젝트의 성과는 높은 수치 달성에 그치지 않고, mAP@0.5와 mAP@0.5:0.95 간의 격차를 분석하고 PR Curve 및 Confusion Matrix를 통해, 모델이 spur 클래스에 취약하고 정밀한 위치 예측 능력이 부족하다는 구체적인 한계를 심층적으로 도출했다는 데 의의가 있습니다.

### 6.2 한계
본 모델은 mAP@0.5:0.95 수치가 0.520으로, 결함의 경계선을 정밀하게 회귀하는 데 명확한 한계를 보였습니다. 이는 PCB 검사에서 결함의 정확한 크기 측정이 중요함을 고려할 때 개선이 필요한 부분입니다. 특히 spur 결함의 저조한 성능과, 이것이 spurious_copper와 시각적으로 혼동되는 문제는 향후 우선적으로 해결해야 할 과제입니다. 향후, spur와 같이 비정형적인 결함 데이터를 추가 확보하고, 특정 클래스에 가중치를 부여하는 손실 함수(예: Focal Loss)를 적용하는 방안을 모색할 수 있습니다. 또한, DFL(Distribution Focal Loss) 파라미터 튜닝이나 고해상도 이미지(1280x1280) 학습을 통해 미세 결함의 특징 포착 능력을 향상시키는 방안을 고려할 수 있습니다.
