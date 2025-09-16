from ultralytics import YOLO

def run_yolov8_finetuning():
    """
    YOLOv8n 모델을 로드하고, 지정된 설정으로 파인튜닝을 수행합니다.
    - 모델: yolov8n.pt (사전 학습된 YOLOv8n 모델)
    - 이미지 크기: 224x224
    - 데이터셋: dataset.yaml에 정의된 경로 사용
    - 에폭: 50 (예시, 필요에 따라 조정 가능)
    - 배치 크기: 16 (예시, GPU 메모리에 따라 조정 가능)
    """
    print("YOLOv8n 모델 파인튜닝을 시작합니다.")

    # YOLOv8n 모델 로드 (사전 학습된 가중치 사용)
    model = YOLO('yolov8n.pt')

    # 모델 파인튜닝
    # imgsz=224: 입력 이미지 크기를 224x224로 설정
    # data='dataset.yaml': 데이터셋 설정 파일 지정
    # epochs=50: 학습 에폭 수
    # batch=16: 배치 크기
    # name='yolov8n_finetune': 학습 결과가 저장될 디렉토리 이름
    results = model.train(data='dataset.yaml', imgsz=224, epochs=50, batch=16, name='yolov8n_finetune')

    print("YOLOv8n 모델 파인튜닝이 완료되었습니다.")
    print(f"학습 결과는 runs/detect/yolov8n_finetune 디렉토리에 저장됩니다.")

if __name__ == '__main__':
    run_yolov8_finetuning()