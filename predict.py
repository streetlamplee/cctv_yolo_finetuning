
import cv2
import argparse
import glob
import os
from ultralytics import YOLO

def find_latest_pt_model():
    """ 가장 최근에 수정된 best.pt 파일을 찾습니다. """
    search_path = 'runs/detect/**/weights/best.pt'
    model_files = glob.glob(search_path, recursive=True)
    
    if not model_files:
        return None
    
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def main(image_path):
    # 1. 모델 경로 설정
    model_path = find_latest_pt_model()
    if not model_path:
        print("오류: .pt 모델 파일(best.pt)을 찾을 수 없습니다.")
        print("main.py를 먼저 실행하여 모델을 훈련했는지 확인하세요.")
        return

    print(f"사용 모델: {model_path}")
    print(f"입력 이미지: {image_path}")

    # 2. YOLO 모델 로드
    try:
        model = YOLO("/home/user/PycharmProjects/AutoEncoderYOLO/src/YOLO/runs/detect/yolov8n_finetune/weights/best.pt")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # 3. 추론 수행
    try:
        results = model(image_path)
    except Exception as e:
        print(f"추론 중 오류 발생: {e}")
        return

    # 4. 결과 시각화
    # results[0].plot()는 결과가 그려진 numpy 배열(BGR)을 반환합니다.
    annotated_frame = results[0].plot()
    
    # 5. 결과 이미지 보여주기
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    print("결과 창이 열렸습니다. 아무 키나 누르면 종료됩니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='YOLOv8 .pt 모델을 사용하여 이미지 추론을 수행합니다.')
    # parser.add_argument('image', type=str, help='추론을 수행할 이미지 파일의 경로')
    # args = parser.parse_args()
    image_path = "./data/images/Test/frames_00010000095000000_00000006.jpg"
    main(image_path)
