"""Video utility functions"""

import subprocess
from pathlib import Path
from typing import Optional


def reencode_video(
    input_path: str,
    output_path: Optional[str] = None,
    codec: str = "libx264",
    crf: int = 23,
    preset: str = "medium",
) -> str:
    """
    Re-encode video to fix corruption issues

    Args:
        input_path: Path to input video
        output_path: Path to output video (optional, will add _reencoded suffix)
        codec: Video codec (default: libx264)
        crf: Constant Rate Factor, 0-51 (lower = better quality, default: 23)
        preset: Encoding speed (ultrafast, fast, medium, slow, veryslow)

    Returns:
        Path to re-encoded video
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = (
            input_path.parent / f"{input_path.stem}_reencoded{input_path.suffix}"
        )
    else:
        output_path = Path(output_path)

    # FFmpeg command for re-encoding
    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-c:a",
        "aac",  # Audio codec
        "-b:a",
        "128k",  # Audio bitrate
        "-y",  # Overwrite output
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Install with: yay -S ffmpeg")


def check_video_integrity(video_path: str) -> dict:
    """
    Check video file integrity using ffprobe

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=nb_read_packets,duration,width,height,r_frame_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        parts = result.stdout.strip().split(",")

        if len(parts) >= 4:
            fps_parts = parts[0].split("/")
            fps = (
                float(fps_parts[0]) / float(fps_parts[1])
                if len(fps_parts) == 2
                else float(fps_parts[0])
            )

            return {
                "fps": fps,
                "width": int(parts[1]),
                "height": int(parts[2]),
                "duration": float(parts[3]) if parts[3] else 0,
                "packets": int(parts[4]) if len(parts) > 4 else 0,
            }
        return {}
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return {}
