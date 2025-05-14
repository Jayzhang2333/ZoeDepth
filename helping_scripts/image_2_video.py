import re
import cv2
from pathlib import Path

def images_to_video(image_folder, output_path, fps=5):
    folder = Path(image_folder)

    # 1. gather and sort by the trailing index (e.g. depth_vis_000 → 0)
    files = list(folder.glob("*.png"))
    def extract_idx(p: Path):
        m = re.search(r"(\d+)(?=\.\w+$)", p.name)
        return int(m.group(1)) if m else -1

    files.sort(key=extract_idx)

    if not files:
        raise ValueError(f"No PNG images found in {image_folder}")

    # 2. read first frame to get size
    first = cv2.imread(str(files[0]))
    h, w = first.shape[:2]

    # 3. setup writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 4. write frames
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"⚠️ Skipping unreadable {p.name}")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)

    # 5. finalize
    writer.release()
    print(f"✅ Video saved to {output_path}")

if __name__ == "__main__":
    images_to_video("/media/jay/Lexar/underwater_depth_videos/canyon/images", "/media/jay/Lexar/underwater_depth_videos/canyon/canyon.mp4", fps=10)
