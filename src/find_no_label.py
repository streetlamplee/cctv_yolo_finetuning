import os
import glob


def find_non_empty_txt_files(search_directory='./raw_data'):
    """
    지정된 디렉토리와 그 하위 디렉토리에서 크기가 1바이트 이상인
    .txt 파일들의 경로를 찾아 리스트로 반환합니다.

    Args:
        search_directory (str): 검색을 시작할 폴더 경로. 기본값은 현재 폴더입니다.

    Returns:
        list: 1바이트 이상인 .txt 파일 경로의 리스트.
    """

    # 1. glob을 사용하여 모든 .txt 파일의 경로를 찾습니다.
    # recursive=True 옵션으로 하위 폴더까지 모두 검색합니다.
    search_pattern = os.path.join(search_directory, '**', '*.txt')
    all_txt_files = glob.glob(search_pattern, recursive=True)

    non_empty_files = []

    # 2. 각 파일의 용량을 확인합니다.
    for file_path in all_txt_files:
        # os.path.getsize()는 파일 크기를 바이트 단위로 반환합니다.
        if os.path.getsize(file_path) >= 1:
            # 3. 파일 크기가 1바이트 이상이면 리스트에 추가합니다.
            non_empty_files.append(file_path)

    return non_empty_files

if __name__ == "__main__":
    # print(find_non_empty_txt_files())
    import shutil
    import random
    li = find_non_empty_txt_files()

    split_ratio = 0.8
    random.shuffle(li)

    to_train = li[:int(len(li) * split_ratio)]
    to_valid = li[int(len(li) * split_ratio):]

    print(int(len(li)* split_ratio))

    for t in to_train:
        t_jpg = t.replace(".txt", ".jpg")
        shutil.move(t_jpg, "/home/user/PycharmProjects/cctv_yolo_finetuning/data/images/train")
        shutil.move(t, "/home/user/PycharmProjects/cctv_yolo_finetuning/data/labels/train")

    for v in to_valid:
        v_jpg = v.replace(".txt", ".jpg")
        shutil.move(v_jpg, "/home/user/PycharmProjects/cctv_yolo_finetuning/data/images/Validation")
        shutil.move(v, "/home/user/PycharmProjects/cctv_yolo_finetuning/data/labels/Validation")


