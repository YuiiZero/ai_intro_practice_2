from pathlib import Path
import subprocess
import shutil

PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_DIR / "videos" / "input.avi"
FRAMES_DIR = PROJECT_DIR / "frames"

FPS = 2

def extract_frames():
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")

    if not VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Video file not found at {VIDEO_PATH}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    for frame in FRAMES_DIR.glob("*.jpg"):
        frame.unlink()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(VIDEO_PATH),
        "-vf", f"fps={FPS}",
        "-q:v", "2",
        str(FRAMES_DIR / "frame_%06d.jpg"),
    ]

    subprocess.run(command, check=True)

    frames_count = len(list(FRAMES_DIR.glob("*.jpg")))

    print("Extraction successful")
    print(f"Video: {VIDEO_PATH}")
    print(f"Frames count: {frames_count} frames")
    print(f"Results saved to: {FRAMES_DIR}")

if __name__ == "__main__":
    extract_frames()