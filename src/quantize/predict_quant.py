import cv2
import numpy as np
import onnxruntime
import yaml
import os
from pathlib import Path

# --- 설정 값 ---
IMG_SIZE = 224
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_YAML_PATH = f'{ROOT_DIR}/data/data.yaml'

def load_class_names(yaml_path):
    """ YAML 파일에서 클래스 이름을 로드합니다. """
    if not os.path.exists(yaml_path):
        print(f"오류: 데이터 YAML 파일을 찾을 수 없습니다: {yaml_path}")
        return None
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def preprocess_image(image_path, size):
    """ 이미지를 읽고 모델 입력에 맞게 비율을 유지하며 전처리(letterbox)합니다. """
    if not os.path.exists(image_path):
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return None, None, None

    img = cv2.imread(image_path)
    original_shape = img.shape[:2]  # (height, width)

    # --- Letterbox --- 
    h, w = original_shape
    r = size / max(h, w)
    new_unpad_w, new_unpad_h = int(w * r), int(h * r)
    
    img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = size - new_unpad_w, size - new_unpad_h
    pad_w, pad_h = pad_w / 2, pad_h / 2  # 양쪽에 패딩 분배

    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # --- End Letterbox ---

    # YOLOv8 스타일 전처리
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(img_transposed, axis=0)  # 배치 차원 추가
    
    scale_info = (r, (left, top))
    
    return input_tensor, img, scale_info

def postprocess_output(output, scale_info, conf_threshold, iou_threshold):
    """ 모델 출력을 후처리하여 바운딩 박스, 점수, 클래스 ID를 추출합니다. """
    output = np.transpose(output[0], (1, 0))
    
    boxes, scores, class_ids = [], [], []
    ratio, pad = scale_info
    pad_x, pad_y = pad

    for row in output:
        class_scores = row[4:]
        max_score = np.max(class_scores)
        
        if max_score > conf_threshold:
            class_id = np.argmax(class_scores)
            cx, cy, w, h = row[:4]
            
            x1 = (cx - w/2 - pad_x) / ratio
            y1 = (cy - h/2 - pad_y) / ratio
            x2 = (cx + w/2 - pad_x) / ratio
            y2 = (cy + h/2 - pad_y) / ratio
            
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(float(max_score))
            class_ids.append(class_id)
            
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return [boxes[i] for i in indices], [scores[i] for i in indices], [class_ids[i] for i in indices]

    return [], [], []

def draw_detections(image, boxes, scores, class_ids, class_names):
    """ 이미지에 탐지된 바운딩 박스와 레이블을 그립니다. """
    for box, score, class_id in zip(boxes, scores, class_ids):
        x, y, w, h = box
        label = f'{class_names[class_id]}: {score:.2f}'
        color = (int(class_id * 50) % 255, int(class_id * 90) % 255, int(class_id * 120) % 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image

def main(model_path, image_path):
    # 1. 클래스 이름 로드
    class_names = load_class_names(DATA_YAML_PATH)
    if class_names is None:
        return

    # 2. 이미지 전처리
    input_tensor, original_image, scale_info = preprocess_image(image_path, IMG_SIZE)
    if input_tensor is None:
        return

    # 3. ONNX 런타임 세션 생성
    try:
        session = onnxruntime.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"ONNX 모델 로드 중 오류 발생: {e}")
        return

    # 4. 추론 실행
    outputs = session.run(None, {input_name: input_tensor})

    # 5. 결과 후처리
    boxes, scores, class_ids = postprocess_output(outputs[0], scale_info, CONF_THRESHOLD, IOU_THRESHOLD)

    # 6. 결과 시각화
    if boxes:
        print(f"{len(boxes)}개의 객체를 탐지했습니다.")
        result_image = draw_detections(original_image, boxes, scores, class_ids, class_names)
    else:
        print("탐지된 객체가 없습니다.")
        result_image = original_image

    # 7. 결과 이미지 보여주기
    cv2.imshow('ONNX Quantized Inference', result_image)
    print("결과 창이 열렸습니다. 아무 키나 누르면 종료됩니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # --- 사용 예시 ---
    # 아래 경로들을 실제 모델과 이미지 경로로 변경하여 사용하세요.
    # model_path = "/home/user/PycharmProjects/cctv_yolo_finetuning/src/best.quant.int8.onnx"
    model_path = "/media/user/F7F0-F779/CCTVapp/resource/yolov8n.quant.onnx"
    for i in os.listdir("/home/user/PycharmProjects/cctv_yolo_finetuning/data/images/Validation"):
        image_path = os.path.join("/home/user/PycharmProjects/cctv_yolo_finetuning/data/images/Validation", i)

        # 파일 존재 여부 확인
        if not os.path.exists(model_path):
            print(f"오류: 모델 파일을 찾을 수 없습니다: {model_path}")
            print("main.py를 실행하여 모델을 먼저 생성하거나, 경로를 올바르게 지정해주세요.")
        elif not os.path.exists(image_path):
            print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
        else:
            # 함수에 직접 인자를 전달하여 호출
            main(model_path, image_path)
