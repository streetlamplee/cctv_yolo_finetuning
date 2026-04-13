"""
split_train_valid.py
─────────────────────────────────────────────────────────────────────────────
output/images/ 와 output/labels/ 의 파일을 랜덤 셔플 후 8:2로 분할합니다.

실행 전 구조:
  output/
  ├── images/  *.jpg
  └── labels/  *.txt

실행 후 구조:
  output/
  ├── images/
  │   ├── train/  *.jpg
  │   └── valid/  *.jpg
  └── labels/
      ├── train/  *.txt
      └── valid/  *.txt

사용법:
  python split_train_valid.py [--base_dir ./data/00_raw_video] [--ratio 0.8] [--seed 42]
"""

import shutil
import random
import argparse
from pathlib import Path


# ───────────────────────────── 설정 ─────────────────────────────

DEFAULT_BASE  = "./data/00_raw_video"
TRAIN_RATIO   = 0.8
RANDOM_SEED   = 42


# ─────────────────────────── 메인 로직 ──────────────────────────

def main():
    parser = argparse.ArgumentParser(description="train/valid 분할")
    parser.add_argument("--base_dir", default=DEFAULT_BASE)
    parser.add_argument("--ratio", type=float, default=TRAIN_RATIO,
                        help="train 비율 (default: 0.8)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="랜덤 시드 (default: 42)")
    args = parser.parse_args()

    out_dir    = Path(args.base_dir) / "output"
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"

    # ── 검사 ──────────────────────────────────────────────────
    if not images_dir.exists():
        raise SystemExit(f"[ERROR] 폴더 없음: {images_dir}")
    if not labels_dir.exists():
        raise SystemExit(f"[ERROR] 폴더 없음: {labels_dir}")

    # ── 파일 목록 수집 (jpg 기준, 대응하는 txt 확인) ──────────
    jpg_files = sorted(images_dir.glob("*.jpg"))

    paired, missing = [], []
    for jpg in jpg_files:
        txt = labels_dir / f"{jpg.stem}.txt"
        if txt.exists():
            paired.append(jpg.stem)
        else:
            missing.append(jpg.stem)

    if missing:
        print(f"[WARN] 라벨 없는 이미지 {len(missing)}개 제외: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    if not paired:
        raise SystemExit("[ERROR] 유효한 이미지-라벨 쌍이 없습니다.")

    # ── 랜덤 셔플 & 분할 ──────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(paired)

    split_idx   = int(len(paired) * args.ratio)
    train_stems = paired[:split_idx]
    valid_stems = paired[split_idx:]

    print(f"\n총 {len(paired)}개  →  train: {len(train_stems)}개 / valid: {len(valid_stems)}개  (seed={args.seed})")

    # ── 출력 폴더 생성 ────────────────────────────────────────
    for split in ["train", "valid"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)

    # ── 파일 이동 ─────────────────────────────────────────────
    def move_pair(stems, split):
        for stem in stems:
            shutil.move(str(images_dir / f"{stem}.jpg"),
                        str(images_dir / split / f"{stem}.jpg"))
            shutil.move(str(labels_dir / f"{stem}.txt"),
                        str(labels_dir / split / f"{stem}.txt"))

    move_pair(train_stems, "train")
    move_pair(valid_stems, "valid")

    # ── 기존 루트 폴더의 남은 파일 정리 (쌍 없는 jpg 등) ─────
    for leftover in images_dir.glob("*.jpg"):
        leftover.unlink()
        print(f"[CLEAN] 쌍 없는 이미지 삭제: {leftover.name}")
    for leftover in labels_dir.glob("*.txt"):
        leftover.unlink()
        print(f"[CLEAN] 쌍 없는 라벨 삭제: {leftover.name}")

    print(f"\n✅ 완료!")
    print(f"   output/images/train : {len(train_stems)}개")
    print(f"   output/images/valid : {len(valid_stems)}개")
    print(f"   output/labels/train : {len(train_stems)}개")
    print(f"   output/labels/valid : {len(valid_stems)}개")


if __name__ == "__main__":
    main()