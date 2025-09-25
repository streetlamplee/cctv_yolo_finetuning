import os
import glob
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_static, CalibrationDataReader


class YOLOv8CalibrationDataReader(CalibrationDataReader):
    """
    ONNX 정적 양자화를 위한 보정 데이터 리더 클래스.
    YAML 파일에 정의된 YOLO 검증 데이터셋을 사용합니다.
    """

    def __init__(self, yaml_path: str, img_size: int = 640, max_samples: int = 3000):
        super().__init__()
        self.img_size = img_size
        self.max_samples = max_samples

        # 1. YAML 파일 로드 및 이미지 경로 확인
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # YAML 파일 위치를 기준으로 상대 경로를 절대 경로로 변환
            base_dir = os.path.dirname(yaml_path)
            # 'val' 또는 'train' 경로 사용 (보통 val 사용)
            image_dir = os.path.join(base_dir, data.get('train') or data.get('val'))

        self.image_files = glob.glob(os.path.join(image_dir, '*.*'))[:self.max_samples]

        if not self.image_files:
            raise ValueError(f"'{image_dir}' 경로에서 이미지를 찾을 수 없습니다. YAML 파일 경로와 내용을 확인하세요.")

        # 2. ONNX 모델의 입력 이름 설정 및 이터레이터 초기화
        self.input_name = 'images'
        self.iterator = iter(self.image_files)
        print(f"총 {len(self.image_files)}개의 보정용 이미지를 찾았습니다.")

    def get_next(self):
        try:
            image_path = next(self.iterator)
            # YOLOv8 Letterbox 전처리 수행
            preprocessed_image = self.preprocess_image(image_path)
            return {self.input_name: preprocessed_image}
        except StopIteration:
            return None  # 모든 데이터를 사용하면 None 반환

    def preprocess_image(self, image_path: str):
        """YOLOv8의 Letterbox 리사이즈 및 전처리를 수행합니다."""
        img = cv2.imread(image_path)

        # Letterbox 리사이즈
        h, w, _ = img.shape
        r = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(w * r), int(h * r)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        top, left = (self.img_size - new_h) // 2, (self.img_size - new_w) // 2
        padded_img[top:top + new_h, left:left + new_w] = resized_img

        # BGR -> RGB, HWC -> CHW, 정규화 및 배치 차원 추가
        rgb_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img.astype(np.float32) / 255.0
        chw_img = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(chw_img, axis=0)

        return input_tensor


if __name__ == '__main__':
    # --- 1. 설정 (사용자 수정 필요) ---
    PT_PATH = 'runs/detect/yolov8n_finetune0/weights/best.pt'  # fine-tuning된 .pt 모델 파일 경로
    YAML_PATH = 'data/data.yaml'  # 데이터셋 .yaml 파일 경로
    IMG_SIZE = 224  # 모델 export 및 양자화 시 사용할 이미지 크기

    # --- 출력 파일 이름 설정 ---
    base_name = os.path.basename(PT_PATH).split('.')[0]
    FP32_ONNX_PATH = f'{base_name}_fp32.onnx'
    INT8_ONNX_PATH = f'{base_name}_int8_quant.onnx'

    # --- 2. PT 모델을 FP32 ONNX로 변환 ---
    print(f"'{PT_PATH}' 모델을 ONNX FP32 형식으로 변환합니다...")
    model = YOLO(PT_PATH)
    # opset 버전을 12로 지정하여 호환성을 높입니다.
    model.export(format='onnx', imgsz=IMG_SIZE, opset=13)
    # export된 파일 이름을 명확하게 변경
    exported_name = PT_PATH.replace('.pt', '.onnx')
    if os.path.exists(exported_name):
        os.rename(exported_name, FP32_ONNX_PATH)
        print(f"ONNX FP32 모델이 '{FP32_ONNX_PATH}'에 저장되었습니다.")
    else:
        raise FileNotFoundError(f"'{exported_name}' 파일이 생성되지 않았습니다. 모델 export에 실패했습니다.")

    # --- 3. FP32 ONNX 모델을 INT8로 정적 양자화 ---
    print("\n정적 양자화를 시작합니다. 보정 데이터셋을 로드합니다...")
    calibration_data_reader = YOLOv8CalibrationDataReader(
        yaml_path=YAML_PATH,
        img_size=IMG_SIZE
    )

    print("ONNX Runtime으로 양자화를 수행합니다 (시간이 다소 걸릴 수 있습니다)...")
    quantize_static(
        model_input=FP32_ONNX_PATH,
        model_output=INT8_ONNX_PATH,
        calibration_data_reader=calibration_data_reader,
        quant_format=onnxruntime.quantization.QuantFormat.QDQ,  # QDQ 형식이 호환성이 좋음
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=['/model.22/Concat_3', '/model.22/Split', '/model.22/Sigmoid'
                                                                   '/model.22/dfl/Reshape', '/model.22/dfl/Transpose',
                          '/model.22/dfl/Softmax',
                          '/model.22/dfl/conv/Conv', '/model.22/dfl/Reshape_1', '/model.22/Slice_1',
                          '/model.22/Slice', '/model.22/Add_1', '/model.22/Sub', '/model.22/Div_1',
                          '/model.22/Concat_4', '/model.22/Mul_2', '/model.22/Concat_5'],
        per_channel=False,
        reduce_range=True,
    )
    print(f"INT8 정적 양자화가 완료되었습니다. 최종 모델이 '{INT8_ONNX_PATH}'에 저장되었습니다.")