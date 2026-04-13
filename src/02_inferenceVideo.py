"""
02_predict_video.py
─────────────────────────────────────────────────────────────────────────────
학습된 best.pt 모델로 영상 추론 후 {filename}_result.mp4로 저장합니다.

사용법:
  python src/02_predict_video.py
  (INPUT_VIDEO_PATH만 아래에서 직접 수정)
"""

from pathlib import Path
import glob
import os
import cv2


# ====================================================================
#  ★ 여기만 수정하세요 ★
# ====================================================================

INPUT_VIDEO_PATH = "/home/user/02_py_project/02_cctv_yolo_finetuning/data/00_raw_video/59.mp4"

CONF  = 0.35   # confidence threshold
IOU   = 0.5    # IoU threshold

# ====================================================================

from ultralytics import YOLO


ROOT_DIR = Path(__file__).parent.parent


def find_latest_pt_model():
    search_path = str(ROOT_DIR / 'runs' / 'detect' / '**' / 'weights' / 'best.pt')
    model_files = glob.glob(search_path, recursive=True)
    if not model_files:
        return None
    return max(model_files, key=os.path.getmtime)


def main():
    # ── 모델 로드 ──────────────────────────────────────────────
    model_path = find_latest_pt_model()
    if not model_path:
        raise SystemExit("[ERROR] best.pt를 찾을 수 없습니다. 먼저 00_main.py로 학습을 실행하세요.")
    print(f"사용 모델  : {model_path}")

    video_path = Path(INPUT_VIDEO_PATH)
    if not video_path.exists():
        raise SystemExit(f"[ERROR] 영상 파일 없음: {video_path}")
    print(f"입력 영상  : {video_path}")

    model = YOLO(model_path)

    # ── 영상 정보 읽기 ─────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── 출력 경로 설정 ─────────────────────────────────────────
    out_path = video_path.parent / f"{video_path.stem}_result.mp4"
    writer   = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    print(f"출력 영상  : {out_path}")
    print(f"해상도     : {width}x{height}  /  FPS: {fps}  /  총 프레임: {total}")
    print("추론 시작...\n")

    # ── 프레임별 추론 ──────────────────────────────────────────
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=CONF, iou=IOU, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  진행: {frame_idx} / {total} 프레임 ({frame_idx/total*100:.1f}%)")

    cap.release()
    writer.release()

    print(f"\n✅ 완료! 결과 저장: {out_path}")


if __name__ == '__main__':
    main()