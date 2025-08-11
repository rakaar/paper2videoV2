#!/usr/bin/env python3
"""
Stitch slide PNGs and per-slide WAV audio files into a single MP4 using ffmpeg.

Usage examples:
  python stitch_video.py --paper-name test_paper
  python stitch_video.py --png-dir artifacts/test_paper/pngs --audio-dir artifacts/test_paper/audio --output artifacts/test_paper/video.mp4

Assumptions:
- PNGs follow Marp naming: deck.001.png, deck.002.png, ...
- Audio files are per slide: slide_001.wav, slide_002.wav, ...
- ffmpeg must be installed and on PATH.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def detect_pairs(png_dir: Path, audio_dir: Path):
    img_regex = re.compile(r"deck\.(\d{3})\.png$")
    aud_regex = re.compile(r"slide_(\d{3})\.wav$")

    imgs = {}
    for p in sorted(png_dir.glob("deck.*.png")):
        m = img_regex.search(p.name)
        if m:
            imgs[m.group(1)] = p

    auds = {}
    for a in sorted(audio_dir.glob("slide_*.wav")):
        m = aud_regex.search(a.name)
        if m:
            auds[m.group(1)] = a

    common = sorted(set(imgs.keys()) & set(auds.keys()))
    pairs = [(k, imgs[k], auds[k]) for k in common]
    return pairs


def ensure_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("Error: ffmpeg not found on PATH.")
        sys.exit(1)
    return ffmpeg


def make_clip(ffmpeg: str, img: Path, aud: Path, out: Path):
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-stats",
        "-loop", "1", "-i", str(img),
        "-i", str(aud),
        "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)


def concat_clips(ffmpeg: str, concat_file: Path, out_mp4: Path):
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-stats",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Stitch PNG slides and WAV audio into an MP4 video")
    ap.add_argument("--paper-name", type=str, default=None, help="Paper name under artifacts/ (e.g., test_paper)")
    ap.add_argument("--png-dir", type=Path, default=None, help="Directory containing deck.XXX.png files")
    ap.add_argument("--audio-dir", type=Path, default=None, help="Directory containing slide_XXX.wav files")
    ap.add_argument("--output", type=Path, default=None, help="Output MP4 path")
    ap.add_argument("--work-dir", type=Path, default=None, help="Temporary work directory for intermediate clips")
    args = ap.parse_args()

    if args.paper_name:
        base = Path("artifacts") / args.paper_name
        png_dir = args.png_dir or (base / "pngs")
        audio_dir = args.audio_dir or (base / "audio")
        output = args.output or (base / "video.mp4")
        work_dir = args.work_dir or (base / "tmp_video")
    else:
        if not (args.png_dir and args.audio_dir and args.output):
            ap.error("Provide either --paper-name or all of --png-dir, --audio-dir, --output")
        png_dir, audio_dir, output = args.png_dir, args.audio_dir, args.output
        work_dir = args.work_dir or (output.parent / "tmp_video")

    png_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    pairs = detect_pairs(png_dir, audio_dir)
    if not pairs:
        print(f"No matching slide pairs found in {png_dir} and {audio_dir}")
        sys.exit(1)

    ffmpeg = ensure_ffmpeg()
    print(f"Found {len(pairs)} slide pairs. Creating per-slide clips in {work_dir} ...")

    concat_path = work_dir / "concat.txt"
    with open(concat_path, "w") as f:
        for idx, img, aud in pairs:
            clip_path = work_dir / f"clip_{idx}.mp4"
            print(f"- Slide {idx}: {img.name} + {aud.name} -> {clip_path.name}")
            make_clip(ffmpeg, img, aud, clip_path)
            f.write(f"file '{clip_path.resolve().as_posix()}'\n")

    print(f"Concatenating {len(pairs)} clips into {output} ...")
    concat_clips(ffmpeg, concat_path, output)
    print(f"Done. Wrote {output}")


if __name__ == "__main__":
    main()
