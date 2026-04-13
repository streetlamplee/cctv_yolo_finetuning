"""
extract_labeled_frames.py
─────────────────────────────────────────────────────────────────────────────
mp4 영상에서 라벨(txt)이 존재하는 프레임만 추출해서
  data/00_raw_video/output/
    ├── images/51_frame_000000.jpg
    ├── images/52_frame_000003.jpg
    ├── labels/51_frame_000000.txt
    └── labels/52_frame_000003.txt
형태로 저장합니다. (video id를 파일명 prefix로 사용, 빈 라벨 제외)

사용법:
  python extract_labeled_frames.py [--base_dir ./data/00_raw_video] [--quality 95]

의존성:
  pip install opencv-python tqdm
"""

import re
import shutil
import argparse
from pathlib import Path

try:
    import cv2
except ImportError:
    raise SystemExit("opencv-python이 없습니다.  pip install opencv-python")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ───────────────────────────── 설정 ─────────────────────────────

DEFAULT_BASE = "./data/00_raw_video"
VIDEO_IDS    = [str(i) for i in range(63, 65)]   # 51 ~ 62
JPEG_QUALITY = 95                                  # 저장 품질 (0‑100)


# ─────────────────────────── 유틸 함수 ──────────────────────────

def find_label_root(labels_dir: Path) -> Path | None:
    """labels/train/ 우선, 없으면 labels/ 바로 아래 탐색."""
    for sub in ["train", "val", "test", ""]:
        candidate = labels_dir / sub if sub else labels_dir
        if candidate.is_dir() and any(candidate.glob("*.txt")):
            return candidate
    return None


def parse_frame_index(stem: str) -> int | None:
    """'frame_000042' → 42  (숫자 없으면 None)"""
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def iter_wrap(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if HAS_TQDM else iterable


# ─────────────────────────── 메인 로직 ──────────────────────────

def process_video(video_id: str, base_dir: Path, out_images: Path,
                  out_labels: Path, quality: int) -> int:

    mp4_path   = base_dir / f"{video_id}.mp4"
    labels_dir = base_dir / video_id / "labels"

    # ── 기본 검사 ─────────────────────────────────────────────
    if not mp4_path.exists():
        print(f"[SKIP] {mp4_path} 없음")
        return 0
    if not labels_dir.exists():
        print(f"[SKIP] {labels_dir} 없음")
        return 0

    label_root = find_label_root(labels_dir)
    if label_root is None:
        print(f"[SKIP] {labels_dir} 안에 txt 파일 없음")
        return 0

    # ── 유효한 라벨 파일 수집 (빈 파일 제외) ─────────────────
    txt_files = sorted(label_root.glob("*.txt"))
    txt_files = [t for t in txt_files if t.stat().st_size > 0]

    if not txt_files:
        print(f"[SKIP] {video_id}: 유효한 라벨 없음 (모두 비어 있음)")
        return 0

    # frame index → txt 경로 매핑
    frame_map: dict[int, Path] = {}
    for t in txt_files:
        idx = parse_frame_index(t.stem)
        if idx is not None:
            frame_map[idx] = t

    if not frame_map:
        print(f"[SKIP] {video_id}: 파일명에서 프레임 번호 파싱 실패")
        return 0

    # ── 영상 열기 ─────────────────────────────────────────────
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"[ERROR] {mp4_path} 열기 실패")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_set   = set(frame_map.keys())
    step = 30
    filtered = [i for i in target_set if i % step == 0]
    print(f"\n[{video_id}] 전체 {total_frames}프레임 | 라벨 있는 프레임: {len(target_set)}개 → {step}배수 필터 후: {len(filtered)}개")
    print(f"         라벨 경로: {label_root}")

    saved     = 0
    frame_idx = 0

    pbar = iter_wrap(range(total_frames), desc=f"  {video_id}", unit="f")

    for _ in pbar:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in target_set and frame_idx % 1 == 0:
            # 파일명: {video_id}_frame_{index:06d}
            stem     = f"{video_id}_frame_{frame_idx:06d}"
            jpg_path = out_images / f"{stem}.jpg"
            txt_dst  = out_labels  / f"{stem}.txt"

            cv2.imwrite(str(jpg_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            shutil.copy2(frame_map[frame_idx], txt_dst)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"  → 저장 완료: {saved}개")
    return saved


def main():
    parser = argparse.ArgumentParser(description="YOLO 라벨 있는 프레임만 추출 (단일 output 폴더)")
    parser.add_argument("--base_dir", default=DEFAULT_BASE,
                        help=f"기본 데이터 경로 (default: {DEFAULT_BASE})")
    parser.add_argument("--quality",  type=int, default=JPEG_QUALITY,
                        help=f"JPEG 품질 0-100 (default: {JPEG_QUALITY})")
    parser.add_argument("--ids", nargs="*", default=VIDEO_IDS,
                        help="처리할 video id 목록 (default: 51~62)")
    args = parser.parse_args()

    base_dir   = Path(args.base_dir)
    out_images = base_dir / "output" / "images"
    out_labels = base_dir / "output" / "labels"

    if not base_dir.exists():
        raise SystemExit(f"[ERROR] base_dir 없음: {base_dir}")

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for vid in args.ids:
        total_saved += process_video(vid, base_dir, out_images, out_labels, args.quality)

    print(f"\n✅ 전체 완료: {total_saved}개 프레임 저장됨")
    print(f"   출력 경로: {base_dir / 'output'}")


if __name__ == "__main__":
    main()