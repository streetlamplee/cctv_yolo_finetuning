import os
import cv2
import numpy as np
import yaml
import glob
from onnxruntime.quantization import CalibrationDataReader

class YOLOv8CalibrationDataReader(CalibrationDataReader):
    """
    ONNX 정적 양자화를 위한 보정 데이터 리더 클래스.
    YAML 파일에 정의된 YOLO 검증 데이터셋을 사용합니다.
    """

    def __init__(self, yaml_path: str, img_size:list, max_samples: int = 500):
        super().__init__()
        self.img_size = img_size
        self.max_samples = max_samples

        # 1. YAML 파일 로드 및 이미지 경로 확인
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # YAML 파일 위치를 기준으로 상대 경로를 절대 경로로 변환
            base_dir = os.path.dirname(yaml_path)
            # 'val' 또는 'train' 경로 사용 (보통 val 사용)
            image_dir = os.path.join(base_dir, data.get('val') or data.get('train'))

        image_files = glob.glob(os.path.join(image_dir, '*.*'))
        count = min(len(image_files), self.max_samples)
        self.image_files = image_files[:count]

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
        img = cv2.resize(img, self.img_size)
        # Letterbox 리사이즈
        # h, w, _ = img.shape
        # r = min(self.img_size / h, self.img_size / w)
        # new_w, new_h = int(w * r), int(h * r)
        # resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        #
        # padded_img = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        # top, left = (self.img_size - new_h) // 2, (self.img_size - new_w) // 2
        # padded_img[top:top + new_h, left:left + new_w] = resized_img

        # BGR -> RGB, HWC -> CHW, 정규화 및 배치 차원 추가
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        normalized_img = rgb_img.astype(np.float32) / 255.0
        chw_img = np.transpose(normalized_img, (2, 0, 1))
        input_tensor = np.expand_dims(chw_img, axis=0)

        return input_tensor
