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
본 연구는 Kaggle의 'PCB Defects' 데이터셋(akhatova/pcb-defects)을 사용하였다. 

이 데이터셋은 6가지 주요 PCB 결함 유형을 포함하고 있다. 각 클래스는 제조 공정상의 각기 다른 문제를 대변한다.
- missing_hole
- mouse_bite
- open_circuit
- short
- spur
- spurious_copper

<img width="1920" height="1920" alt="image" src="https://github.com/user-attachments/assets/f3a4ca0d-3739-4a20-b253-78a2b137826b" />

이 중 spur(회로에서 뾰족하게 튀어나온 돌기)와 spurious_copper(회로와 무관하게 잔류하는 구리 조각)는 모두 '불필요한 구리'라는 시각적 유사성을 공유하며, 형태가 비정형적이고 크기가 미세하여 모델이 탐지하고 분류하기에 가장 까다로운 유형으로 분석되었다.

***

### 2.2 데이터 전처리
- XML 파싱: 수백 개의 개별 어노테이션 파일(.xml)을 파싱하여, 파일명, 원본 이미지 크기(width, height), 결함 클래스, 바운딩 박스 좌표(xmin, ymin, xmax, ymax) 정보를 추출하였다. 이 모든 정보를 단일 Pandas DataFrame으로 변환하여 데이터 관리를 용이하게 하였다.
- 이미지 리사이즈: YOLOv11 모델의 표준 입력 크기이자, 연산 효율성과 탐지 성능 간의 균형을 고려하여 모든 이미지를 640x640 픽셀로 일괄 리사이즈하였다.
좌표 스케일링: 원본 이미지 크기에 맞춰져 있던 바운딩 박스 좌표를, 2단계에서 리사이즈된 640x640 크기에 맞게 비례적으로 스케일링하였다.
- YOLO 포맷 변환: YOLO 학습에 필요한 텍스트(.txt) 라벨 형식(class_id, x_center, y_center, width, height - 모든 값은 0과 1 사이로 정규화됨)으로 변환하였다.
- 데이터 분할: 전체 데이터셋을 학습(Train), 검증(Validation), 테스트(Test) 셋으로 8:1:1의 비율로 명확히 분리하여 저장하였다. 테스트 셋은 모델의 최종 성능 평가를 위해서만 사용되며, 학습 및 검증 과정에는 일절 관여하지 않도록 하여 평가의 객관성을 확보하였다.

***

### 2.3 데이터 증강
모델의 일반화 성능과 강인성을 향상시키기 위해 다양한 데이터 증강 기법을 적용하였다. 
- Mixup(두 이미지 혼합) 및 HSV(색상, 채도, 명도) 변환 : 실제 현장에서 발생할 수 있는 다양한 조명 변화나 카메라 노이즈 환경에 대응할 수 있게 한다.
- Flip(좌우 반전) 및 Degrees(회전) : PCB가 검사 라인에 놓이는 각도가 미세하게 비틀어지더라도 모델이 일관된 성능을 내도록 유도하였다.

이러한 증강 기법은 한정된 데이터셋으로도 수만 장의 데이터를 학습하는 것과 유사한 효과를 내어 과적합을 방지한다.


## 📌 3. Methodology
PCB 결함은 제품의 신뢰도와 직결되므로 신속하고 정확한 검출이 필수적이다. 따라서, 실시간성과 높은 정확도를 강점으로 하는 YOLO(You Only Look Once) 아키텍처를 핵심 방법론으로 튜닝하는 방식을 채택하였다.
### 3.1 데이터셋 분할
전체 데이터셋은 총 757개의 이미지와 4,662개의 결함 라벨을 포함한다. 모델의 학습, 검증, 평가를 위해 데이터셋은 임의로 다음과 같은 비율로 분리하였습니다.
- 학습(Train) 세트: 550개 이미지 (3,393개 라벨)
- 검증(Validation) 세트: 138개 이미지 (846개 라벨)
- 테스트(Test) 세트: 69개 이미지 (423개 라벨)
분할 구조를 통해 모델이 충분한 양의 데이터로 학습하고, 학습 과정에서 과적합(overfitting)을 모니터링하며, 최종적으로 학습에 사용되지 않은 데이터를 통해 일반화 성능을 객관적으로 평가할 수 있도록 설계하였습니다.

***

### 3.2 모델 아키텍처 정의
본 프로젝트는 ultralytics 라이브러리에서 제공하는 YOLO v11 모델 아키텍처의 'small' (s) 버전을 기반으로 한 커스텀 모델로, 속도와 정확도 간의 균형이 뛰어나 실제 산업 현장의 실시간 검사 시스템에 적용하기에 적합하다고 판단하여 선정하였습니다.
- YOLO v11s는 경량화된(small) 버전으로, 백본(Backbone)과 넥(Neck), 헤드(Head)로 구성됩니다.
- CNN과 유사하게, 이미지의 특징(feature)을 추출하는 Convolutional 레이어들이 백본을 이룹니다.
- PANet (Path Aggregation Network) 등의 구조(Neck)를 통해 다양한 스케일의 특징을 집계합니다.
- 최종적으로 헤드에서 바운딩 박스(좌표)와 클래스 확률을 예측합니다.

***

### 3.3 하이퍼파라미터
- batch_size: 16
- learning_rate: 0.001
- epochs: 100
- optimizer: Adam
- image_size: 640

***

### 3.4 학습 프로세스
모델은 train 및 validation 데이터셋을 사용하여 학습하였습니다. 학습은 사전 훈련된(pre-trained) 가중치를 기반으로 전이 학습(Transfer Learning)을 수행하여, 적은 데이터로도 PCB 결함이라는 특정 도메인에 대한 높은 탐지 성능을 높였습니다. 학습 과정 동안 매 에포크(epoch)마다 validation 세트에 대한 성능(예: mAP@0.5)이 측정되었으며, 제공된 노트북에서는 이 측정값 중 가장 높은 성능을 기록한 시점의 모델 가중치가 적용된 best.pt 파일을 최종 모델로 선정하였습니다.

***

### 3.5 모델 성능 평가 (테스트)
로드된 모델의 predict() 또는 val() 메소드를 test 데이터셋에 대해 실행하였습니다. 이 과정에서 모델은 test 이미지를 입력받아 결함을 예측하고, 이 예측 결과(바운딩 박스, 클래스, 신뢰도 점수)를 실제 정답 라벨과 비교합니다. 평가가 완료되면, 객체 탐지 모델의 표준 성능 지표인 mAP가 산출된다. 구체적으로, IoU(Intersection over Union) 임계값에 따른 mAP@0.5 (느슨한 기준) 및 mAP@0.5:0.95 (엄격한 기준) 값을 통해 모델의 정확도를 정량적으로 평가합니다. 이와 더불어 정밀도(Precision)와 재현율(Recall)을 확인하여, 모델이 결함을 놓치지 않고(높은 재현율) 정확하게 예측하는지(높은 정밀도)를 종합적으로 분석하였습니다.

***

### 3.6 시각화 및 디버깅
- 학습 곡선: 학습 과정 중 results.csv 파일이 생성되며, 이를 matplotlib 등으로 시각화하여 에포크별 학습 손실(train loss), 검증 손실(val loss), mAP, 정밀도, 재현율 등의 변화를 추적합니다.-
- 결과 시각화: predict() 실행 시, 원본 이미지에 예측된 바운딩 박스와 신뢰도 점수가 그려진 결과 이미지가 저장되어 직관적인 성능 확인이 가능합니다.
- 성능 지표 플롯: Confusion Matrix(혼동 행렬), P-R Curve(정밀도-재현율 곡선) 등이 자동으로 생성되어 모델의 세부 성능을 분석합니다.

***

### 3.7 구현 및 실행


## 📌 4. Evaluation & Analysis
### 4.1 Train loss / Validation loss / Precision & Recall / mAP
- Train loss : train/box_loss (바운딩 박스), train/cls_loss (분류), train/dfl_loss (분포 초점 손실) 모두 epoch가 진행됨에 따라 부드럽게 감소하며 안정적인 값으로 수렴함.
- Validation loss : val/box_loss, val/cls_loss, val/dfl_loss 또한 Train loss와 유사한 추세로 감소하고 있다. epoch 후반부에 검증 손실이 다시 증가하는 과적합 현상이 보이지 않는 걸로 미루어 보았을 때 모델이 학습 데이터에만 편향되지 않았음을 유추해볼 수 있다.
- Precision & Recall : metrics/precision(B)와 metrics/recall(B) 모두 학습 초기에 빠르게 상승하여 각각 0.95 이상의 매우 높은 수준에서 안정화되었다.
- mAP : metrics/mAP50(B)는 IoU(Intersection over Union) 임계값을 0.5로 설정했을 때의 평균 정밀도이다. 약 0.95에 육박하는 매우 높은 수치를 기록했다.
- mAP : metrics/mAP50-95(B)는 IoU 임계값을 0.5부터 0.95까지 0.05 간격으로 변경하며 측정한 mAP의 평균값이다. 약 0.5 정도에서 수렴했다.
<img width="2400" height="1200" alt="KakaoTalk_20251107_225911826_07" src="https://github.com/user-attachments/assets/144fa164-a2f5-473a-95f1-8a8647c1f647" />


***

### 4.2 Normalized Confusion Matrix
- missing_hole : 100%
- spurious_copper : 96%
- short : 93%
- spur : 85%
- mouse_bite / open_circuit : 81%
- background : 0%
주목할 만한 점은 background 클래스를 예측을 못한다는 것인데, 논의 결과 학습 데이터의 어노테이션 방식에 근본적인 원인이 있는 것으로 판단하였다.
바운딩 박스 방식은 모델에게 사각형 내부의 모든 픽셀 정보는 이 클래스에 속한다라고 학습시킨다는 원리이다. 이로 인해, 비정형적이거나 크기가 작은 결함의 바운딩 박스 내부에 포함된 대다수의 정상 픽셀이 실제 결함의 특징으로 함께 학습되기에 오류가 크게 나는 것 같다.
<img width="3000" height="2250" alt="KakaoTalk_20251107_225911826_04" src="https://github.com/user-attachments/assets/c79a6e1d-7162-4dd8-88bc-5dada2d9ebb3" />

*** 

### 4.3 Test set 최종 평가
학습 및 검증 과정에 사용되지 않은 별도의 테스트 셋(Test Set)으로 모델의 최종 성능을 평가하였다.
- 정량 평가 (성능 표)
: 학습에 사용되지 않은 테스트 셋을 이용한 최종 평가에서, 모델은 전체 All 클래스 기준 정밀도(P) 0.967, 재현율(R) 0.939, mAP@0.5 0.973을 기록하며 전반적으로 높은 수준의 탐지 성능을 입증했다. 그러나 mAP@0.5:0.95는 0.520으로, 학습 과정에서 관찰된 '정밀한 위치 예측의 한계'가 테스트 셋에서도 동일하게 재현됨을 확인하였다. 클래스별 상세 분석을 통해, missing_hole(0.592)과 같이 형태가 명확한 결함 유형은 높은 mAP를 달성한 반면, spur(0.445) 결함은 가장 저조한 mAP@0.5:0.95 점수를 기록한 것을 확인했다. 이는 모델이 비정형적이고 미세한 spur 결함을 정밀하게 예측하는 데 가장 어려움을 겪고 있음을 의미한다.

- 정성 평가 (Confusion Matrix)
: Confusion Matrix(혼동 행렬) 분석 결과, 대부분의 결함은 대각선(True Positive)에 밀집되어 모델이 클래스 분류를 정확히 수행했음을 보여주었다. 하지만 일부 spur 결함이 short 또는 spurious_copper로 오분류되는 사례가 관찰되었다. 이는 spur와 spurious_copper가 모두 '불필요한 구리 조각'이라는 시각적 유사성을 공유하기 때문에 모델이 혼동을 일으킨 것으로 분석된다.

- PR Curve 분석
: Precision-Recall 곡선(PR Curve) 분석 결과 spur 결함의 곡선이 다른 클래스(예: missing_hole)의 곡선보다 확연히 아래쪽에 위치하였다. 이는 spur 결함을 더 많이 찾으려고 할수록(재현율을 높이려 할수록), 관련 없는 것을 spur로 잘못 예측하는(정밀도가 급격히 하락하는) 경향이 다른 결함보다 크다는 것을 의미하며, 성능 표의 spur 결함에 대한 낮은 mAP 수치를 뒷받침한다.
<img width="691" height="528" alt="469852e4-9ecb-42ad-a2e6-87ed9d45bbbf" src="https://github.com/user-attachments/assets/fe0a643d-e814-4d5f-b7e1-427395dcb9ef" />
<img width="1189" height="690" alt="0c622085-535e-454e-b205-a51a97ec4c9d" src="https://github.com/user-attachments/assets/7b6733ec-6624-4784-baae-3748f0158e60" />
<img width="1109" height="989" alt="image" src="https://github.com/user-attachments/assets/9b246606-4938-4f74-99ea-e2ec903c0c07" />
<img width="1189" height="690" alt="81e3fb1e-216d-4492-8955-1bde89ba67ae" src="https://github.com/user-attachments/assets/5341a4d6-c342-4c35-b2e1-c4d8873786cf" />

## 📌 5. Related Works
* J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?," in Advances in Neural Information Processing Systems (NIPS), 2014.
* Ultralytics, "YOLOv11 Documentation.
* J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," in CVPR, 2016.
* I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization (AdamW)," in ICLR, 2019.

## 📌 6. Conclusion
