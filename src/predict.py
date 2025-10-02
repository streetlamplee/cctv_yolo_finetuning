from pathlib import Path
import cv2
import argparse
import glob
import os
from ultralytics import YOLO

def find_latest_pt_model(ROOT_DIR = Path(__file__).parent.parent):
    """ 가장 최근에 수정된 best.pt 파일을 찾습니다. """
    search_path = f'{ROOT_DIR}/runs/detect/**/weights/best.pt'
    model_files = glob.glob(search_path, recursive=True)
    
    if not model_files:
        return None
    
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def main(image_path, show = False):
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
        model = YOLO(model_path)
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # 3. 추론 수행
    try:
        results = model(image_path)
    except Exception as e:
        print(f"추론 중 오류 발생: {e}")
        return

    # 4. 결과 텍스트 파일로 저장
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    with open(txt_path, 'w') as f:
        # .boxes 속성에서 클래스 ID와 정규화된 좌표를 가져옵니다.
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            # xywhn은 [x_center, y_center, width, height] 형식의 정규화된 텐서입니다.
            x, y, w, h = box.xywhn[0]
            line = f"{class_id} {x.item()} {y.item()} {w.item()} {h.item()}\n"
            f.write(line)
    print(f"결과가 '{txt_path}' 파일에 저장되었습니다.")

    # 5. 결과 시각화
    # results[0].plot()는 결과가 그려진 numpy 배열(BGR)을 반환합니다.
    if show:
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
    # image_path = "/home/user/PycharmProjects/cctv_yolo_finetuning/2025_09_19__13_05_44.jpg"
    ROOT_DIR = Path(__file__).parent.parent
    data_folder = os.listdir(f"{ROOT_DIR}/1002_data/raw_data")
    image_folder = [os.path.join(ROOT_DIR, "1002_data", "raw_data", x) for x in data_folder if x.endswith(".jpg")]
    for image in image_folder:
        main(image)
