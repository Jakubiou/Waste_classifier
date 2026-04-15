import os
import cv2
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "own")

CATEGORIES = ["plastic", "paper", "glass", "bio", "mixed"]

FRAME_INTERVAL = 8
IMG_SIZE = 512


def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Nelze otevřít: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"    {total_frames} snímků, {fps:.0f} fps → ~{total_frames // FRAME_INTERVAL} fotek")

    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            out_path = os.path.join(output_dir, f"{video_name}_f{frame_idx:05d}.jpg")
            img.save(out_path, "JPEG", quality=90)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():

    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        for cat in CATEGORIES:
            os.makedirs(os.path.join(VIDEO_DIR, cat), exist_ok=True)
        print(f"\nSložka {VIDEO_DIR} vytvořena.")
        print("Nahraj videa do data/videos/{{kategorie}}/")
        print("Pak spusť znovu.")
        return

    total = 0
    for cat in CATEGORIES:
        cat_video_dir = os.path.join(VIDEO_DIR, cat)
        cat_output_dir = os.path.join(OUTPUT_DIR, cat)

        if not os.path.exists(cat_video_dir):
            continue

        videos = [f for f in os.listdir(cat_video_dir)
                  if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

        if not videos:
            continue

        print(f"\n[{cat.upper()}] – {len(videos)} videí")
        cat_total = 0
        for v in videos:
            print(f"  {v}")
            n = extract_frames(os.path.join(cat_video_dir, v), cat_output_dir)
            print(f"    → {n} fotek uloženo")
            cat_total += n

        print(f"  {cat}: {cat_total} fotek celkem")
        total += cat_total

    print(f"\nHotovo!{total} fotek v {OUTPUT_DIR}")


if __name__ == "__main__":
    main()