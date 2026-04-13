import os
from pathlib import Path
from ultralytics import YOLO
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_static, CalibrationDataReader
from src.quantize.yoloCalibDataset import YOLOv8CalibrationDataReader

ROOT_DIR = Path(__file__).parent.parent.parent

if __name__ == '__main__':
    # --- 1. 설정 (사용자 수정 필요) ---
    PT_PATH = '/home/user/PycharmProjects/cctv_yolo_finetuning/src/best.pt'  # fine-tuning된 .pt 모델 파일 경로
    YAML_PATH = os.path.join(ROOT_DIR, 'data/data.yaml')  # 데이터셋 .yaml 파일 경로
    IMG_SIZE = 224  # 모델 export 및 양자화 시 사용할 이미지 크기

    # --- 출력 파일 이름 설정 ---
    base_name = os.path.basename(PT_PATH).split('.')[0]
    FP32_ONNX_PATH = f'{Path(__file__).parent}/best_fp32.onnx'
    INT8_ONNX_PATH = f'{Path(__file__).parent}/best_int8_quant.onnx'

    # --- 2. PT 모델을 FP32 ONNX로 변환 ---
    print(f"'{PT_PATH}' 모델을 ONNX FP32 형식으로 변환합니다...")
    model = YOLO(PT_PATH)

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