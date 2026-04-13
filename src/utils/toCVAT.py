import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil
import random
from pathlib import Path
from src.predict import main as predict

ROOT_DIR = Path(__file__).parent.parent.parent

def create_cvat_xml(image_dir, label_dir, class_names, output_xml):
    """
    YOLO 형식의 라벨을 CVAT XML 형식으로 변환합니다.

    Args:
        image_dir (str): 원본 이미지 파일이 있는 디렉토리 경로.
        label_dir (str): YOLO 형식의 .txt 라벨 파일이 있는 디렉토리 경로.
        class_names (list): 클래스 이름 목록 (class_id 순서와 일치해야 함).
        output_xml (str): 생성될 CVAT XML 파일의 경로.
    """
    # XML 루트 요소 생성
    annotations = ET.Element('annotations')
    ET.SubElement(annotations, 'version').text = '1.1'

    image_id_counter = 0

    # 라벨 디렉토리의 모든 파일을 순회
    for label_filename in sorted(os.listdir(label_dir)):
        if not label_filename.endswith('.txt'):
            continue

        base_name = os.path.splitext(label_filename)[0]

        # 해당하는 이미지 파일 찾기 (jpg, png 등 확장자 고려)
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break

        if not image_path:
            print(f"경고: 라벨 파일 '{label_filename}'에 해당하는 이미지를 찾을 수 없습니다. 건너뜁니다.")
            continue

        # 이미지 크기 가져오기
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise IOError("이미지를 불러올 수 없습니다.")
            img_height, img_width, _ = image.shape
        except Exception as e:
            print(f"에러: 이미지 파일 '{image_path}'을(를) 읽는 중 오류 발생: {e}. 건너뜁니다.")
            continue

        # <image> 요소 생성
        image_element = ET.SubElement(annotations, 'image', {
            'id': str(image_id_counter),
            'name': os.path.basename(image_path),
            'width': str(img_width),
            'height': str(img_height)
        })

        # YOLO 라벨 파일 읽기
        with open(os.path.join(label_dir, label_filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, width, height = map(float, parts)
                class_id = int(class_id)

                # YOLO 좌표 -> 실제 픽셀 좌표로 변환
                abs_x_center = x_center * img_width
                abs_y_center = y_center * img_height
                abs_width = width * img_width
                abs_height = height * img_height

                # 중심점/너비/높이 -> 좌상단/우하단 좌표로 변환
                xtl = abs_x_center - (abs_width / 2)
                ytl = abs_y_center - (abs_height / 2)
                xbr = abs_x_center + (abs_width / 2)
                ybr = abs_y_center + (abs_height / 2)

                # <box> 요소 생성
                ET.SubElement(image_element, 'box', {
                    'label': class_names[class_id],
                    'occluded': '0',
                    'source': 'manual',
                    'xtl': f"{xtl:.2f}",
                    'ytl': f"{ytl:.2f}",
                    'xbr': f"{xbr:.2f}",
                    'ybr': f"{ybr:.2f}",
                    'z_order': '0'
                })

        image_id_counter += 1

    # 예쁘게 들여쓰기 된 XML 문자열 생성
    xml_str = ET.tostring(annotations, 'utf-8')
    parsed_str = minidom.parseString(xml_str)
    pretty_xml_str = parsed_str.toprettyxml(indent="  ")

    # 파일에 저장
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(pretty_xml_str)

    print(f"변환 완료! 총 {image_id_counter}개의 이미지에 대한 라벨이 '{output_xml}' 파일에 저장되었습니다.")

def make_data_set(raw_data_folder:str, iter:int = 100):
    os.makedirs(os.path.join(ROOT_DIR, "data/need_check/images"), exist_ok=False)
    os.makedirs(os.path.join(ROOT_DIR, "data/need_check/labels"), exist_ok=False)
    image_list = os.listdir(os.path.join(ROOT_DIR, raw_data_folder))
    image_name_list = [x for x in image_list if x.endswith(".jpg")]
    image_list = [os.path.join(ROOT_DIR, raw_data_folder, x) for x in image_list if x.endswith(".jpg")]
    for _ in range(iter):
        i = random.randint(0, len(image_list))
        image_name = image_name_list[i]
        label_name = image_name.replace(".jpg",".txt")
        image_path = image_list[i]
        label_path = image_path.replace(".jpg", ".txt")
        predict(image_path)
        if os.path.exists(image_path):
            shutil.move(image_path, os.path.join(ROOT_DIR, "data/need_check/images", image_name))
            if os.path.exists(label_path):
                shutil.move(label_path, os.path.join(ROOT_DIR, "data/need_check/labels", label_name))



if __name__ == '__main__':
    #
    # list = os.listdir("/home/user/PycharmProjects/cctv_yolo_finetuning/data/need_check")
    # for l in list:
    #     if l.endswith(".jpg"):
    #         shutil.move(os.path.join("/home/user/PycharmProjects/cctv_yolo_finetuning/data/need_check", l), os.path.join("/home/user/PycharmProjects/cctv_yolo_finetuning/data/need_check", "images", l))
    #     elif l.endswith(".txt"):
    #         shutil.move(os.path.join("/home/user/PycharmProjects/cctv_yolo_finetuning/data/need_check", l), os.path.join("/home/user/PycharmProjects/cctv_yolo_finetuning/data/need_check", "labels", l))

    make_data_set("1121_data", 100)

    # --- 사용자 설정 영역 ---

    # 1. 이미지와 라벨이 있는 폴더 경로를 지정하세요.
    image_dir = '/home/user/PycharmProjects/cctv_yolo_finetuning/data/02_interim/images'  # 예: './dataset/images'
    label_dir = '/home/user/PycharmProjects/cctv_yolo_finetuning/data/02_interim/labels'  # 예: './dataset/labels'

    # 2. 클래스 이름을 순서대로(class_id 0부터) 입력하세요.
    class_names = ['standing',               # 0
        'lying down on bed',      # 1
        'sitting on bed',         # 2
        'fallen down',            # 3
        'wheel chair',            # 4
        'unknown status',         # 5
        'sitting on chair',       # 6
        'sitting on the floor',   # 7
        'food tray',              # 8
        'perch on bed',           # 9
        'staff']                  # 10


    # 3. 결과로 나올 XML 파일의 이름을 지정하세요.
    output_xml = os.path.join(ROOT_DIR, 'data/need_check/annotations.xml')

    # --- 스크립트 실행 ---
    create_cvat_xml(image_dir, label_dir, class_names, output_xml)