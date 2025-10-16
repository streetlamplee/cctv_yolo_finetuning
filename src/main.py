from ultralytics import YOLO
import os
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_static, CalibrationDataReader
import yaml
import cv2
import numpy as np
import glob
from src.quantize.yoloCalibDataset import YOLOv8CalibrationDataReader




class YOLOv8DataReader(CalibrationDataReader):
    """
    ONNX 모델 양자화를 위한 보정 데이터 리더 클래스입니다.
    data.yaml 파일에 명시된 검증 데이터셋을 사용합니다.
    """

    def __init__(self, data_yaml_path, batch_size=1, img_size=224):
        self.img_size = img_size
        self.batch_size = batch_size

        # data.yaml 파일 로드하여 검증 이미지 경로 확인
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            # data.yaml 파일의 디렉토리를 기준으로 상대 경로를 절대 경로로 변환
            base_dir = os.path.dirname(data_yaml_path)
            val_path = os.path.join(base_dir, data['val'])
            # glob를 사용하여 모든 이미지 파일 검색 (jpg, png 등)
            self.image_files = glob.glob(os.path.join(val_path, '*.*'))

        self.data_count = len(self.image_files)
        self.enumerator = iter(range(self.data_count))
        self.input_name = 'images'  # YOLOv8 모델의 입력 이름

    def get_next(self):
        try:
            # 다음 이미지 인덱스 가져오기
            idx = next(self.enumerator)
            image_path = self.image_files[idx]

            # 이미지 전처리 (YOLOv8 스타일에 맞게)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            img = cv2.resize(img, (self.img_size, self.img_size))  # 리사이즈
            img = img.astype(np.float32) / 255.0  # 0-1 정규화
            img = np.transpose(img, (2, 0, 1))  # HWC to CHW
            img = np.expand_dims(img, axis=0)  # 배치 차원 추가 [1, 3, H, W]

            # 모델 입력 형식에 맞게 딕셔너리로 반환
            return {self.input_name: img}

        except StopIteration:
            # 모든 데이터를 사용했으면 None 반환
            return None


def quantize_onnx_model(onnx_model_path, data_yaml_path, img_size=224):
    """
    FP32 ONNX 모델을 INT8 정적 양자화합니다.

    Args:
        onnx_model_path (str): 원본 ONNX 모델 파일 경로.
        data_yaml_path (str): 데이터셋 설정 파일 경로.
        img_size (int): 모델 입력 이미지 크기.

    Returns:
        str: 양자화된 ONNX 모델 파일 경로.
    """
    print("INT8 양자화를 시작합니다...")

    # 양자화된 모델이 저장될 경로 설정
    quantized_output_path = onnx_model_path.replace('.onnx', '.quant.int8.onnx')

    # 보정 데이터 리더 생성
    # calibration_data_reader = YOLOv8DataReader(data_yaml_path, img_size=img_size)
    calibration_data_reader = YOLOv8CalibrationDataReader(
        yaml_path=data_yaml_path,
        img_size=img_size
    )

    print("ONNX Runtime으로 양자화를 수행합니다 (시간이 다소 걸릴 수 있습니다)...")
    quantize_static(
        model_input=onnx_model_path,
        model_output=quantized_output_path,
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
    # # 정적 양자화 실행
    # quantize_static(
    #     model_input=onnx_model_path,
    #     model_output=quantized_output_path,
    #     calibration_data_reader=calibration_data_reader,
    #     quant_format=onnxruntime.quantization.QuantFormat.QDQ,  # QDQ는 정확도, QOperator는 성능에 유리
    #     activation_type=QuantType.QInt8,
    #     weight_type=QuantType.QInt8,
    #     per_channel=True,
    #     reduce_range=True,
    #     nodes_to_exclude=[],
    # )

    print(f"INT8 양자화가 완료되었습니다. 모델이 '{quantized_output_path}'에 저장되었습니다.")
    return quantized_output_path


def run_yolov8_finetuning(data_yaml='./data/data.yaml'):
    """
    YOLOv8n 모델을 로드하고, 파인튜닝, ONNX 변환, INT8 양자화를 순차적으로 수행합니다.

    Args:
        data_yaml (str): 데이터셋 설정 파일 경로

    Returns:
        tuple: (best_pth_path, onnx_path, quantized_onnx_path)
               - best_pth_path: 가장 성능이 좋은 .pt 파일 경로 (Pytorch 모델)
               - onnx_path: ONNX 형식으로 변환된 모델 파일 경로
               - quantized_onnx_path: INT8 양자화된 ONNX 모델 파일 경로
    """
    print("YOLOv8n 모델 파인튜닝을 시작합니다.")

    # YOLOv8n 모델 로드
    model = YOLO('yolov8n.pt')

    # 모델 파인튜닝
    results = model.train(data=data_yaml, imgsz=224, epochs=1, batch=16, project='../runs/detect', name='yolov8n_finetune', exist_ok=True)
    print("YOLOv8n 모델 파인튜닝이 완료되었습니다.")

    # 가장 성능 좋은 모델(.pt) 경로 저장
    best_pth_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    print(f"가장 성능이 좋은 모델(best.pt)이 '{best_pth_path}'에 저장되었습니다.")

    # ONNX로 변환
    print("ONNX 형식으로 모델을 내보냅니다...")
    best_model = YOLO(best_pth_path)
    onnx_path = best_model.export(format='onnx')
    print(f"ONNX 모델이 '{onnx_path}'에 저장되었습니다.")

    # ONNX 모델 INT8 양자화 수행
    quantized_onnx_path = quantize_onnx_model(onnx_path, data_yaml, img_size=224)

    return best_pth_path, onnx_path, quantized_onnx_path


if __name__ == '__main__':
    DATASET_YAML_PATH = '../data/data.yaml'

    # 함수로부터 세 개의 파일 경로를 반환받습니다.
    best_model_path, onnx_model_path, quantized_model_path = run_yolov8_finetuning(data_yaml=DATASET_YAML_PATH)

    print("\n--- 최종 결과 ---")
    print(f"Best PTH (PyTorch Model) Path: {best_model_path}")
    print(f"Exported FP32 ONNX Model Path: {onnx_model_path}")
    print(f"Exported INT8 ONNX Model Path: {quantized_model_path}")