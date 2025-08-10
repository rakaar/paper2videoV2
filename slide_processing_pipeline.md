# Slide Processing Pipeline

This document explains the three main components of the slide processing pipeline:
1. Converting slides JSON to Marp slides and saving as PNG files
2. Generating audio files from slide content using Sarvam API
3. Stitching PNG and audio files to create a video using ffmpeg

## 1. Converting Slides JSON to Marp Slides and Saving as PNG Files

### Files Involved
- `json2marp.py` - Converts JSON to Marp Markdown
- `processors/slide_renderer.py` - Renders Marp Markdown to PNG frames

### Key Code

#### json2marp.py
This script converts Gemini-style JSON to Marp Markdown with MathJax enabled.

```python
FRONT_MATTER = """---
marp: true
math: mathjax
paginate: true
theme: gaia
style: |
  /* Global slide tweaks */
  section {
    padding-top: 0.2em;
  }
  section h1 {
    font-size: 1.6em;
    line-height: 1.2;
  }
  /* Ensure images fit within slide without being cut */
  section img {
    max-height: 45vh;
    max-width: 80%;
    height: auto;
    object-fit: contain;
    display: block;
    margin: 1em auto;
  }

  /* When slide has an image, shrink heading and body font */
  section.has-image h1 {
    font-size: 1.2em;
  }
  section.has-image h2 {
    font-size: 1.2em;
  }
  section.has-image ul,
  section.has-image p {
    font-size: 0.8em;
  }
---"""

def main():
    # Parse arguments
    data = json.loads(args.json_file.read_text())
    if isinstance(data, dict) and 'slides' in data:
        slides = data['slides']
    else:
        slides = data
    
    # Process slides
    sorted_slides = sorted(slides, key=lambda x: get_slide_num(x) if get_slide_num(x) is not None else float('inf'))

    slide_markdowns = []
    for s in sorted_slides:
        slide_num = get_slide_num(s) or "N/A"
        title = s.get("title", f"Untitled Slide {slide_num}")
        content = s.get("content", "No content provided.")

        if isinstance(content, list):
            content = "\n".join(content)

        slide_md = f"# {title}\n\n{content}"
        slide_markdowns.append(slide_md)

    all_slides_md = "\n\n---\n\n".join(slide_markdowns)
    final_md = f"{FRONT_MATTER}\n\n<!-- -->\n\n{all_slides_md}"

    args.out.write_text(final_md)
```

#### processors/slide_renderer.py
This script renders Marp markdown as PNG images using marp-cli.

```python
def render_slides(marp_md_path, frames_dir=None):
    # Determine output directory for frames
    if frames_dir is None:
        frames_dir = Path("slides") / "frames"
    else:
        frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Marp CLI needs a dummy file path in the target directory for image sequence output
    frames_output_path_template = frames_dir / "deck.png"
    
    try:
        # Check if marp-cli is installed
        try:
            subprocess.run(["npx", "marp", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            raise RuntimeError("marp-cli not found. Please install it with 'npm i -g @marp-team/marp-cli'")
        
        # Render Marp markdown to PNG frames
        cmd = [
            "npx", "marp", 
            str(marp_md_path),
            "--images", "png",
            "--image-scale", "2",
            "--allow-local-files",
            # Provide a dummy file path; marp-cli will use its basename for the sequence
            "--output", str(frames_output_path_template)
        ]
        
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Error handling code...
            pass
        
        # Verify that frames were generated
        png_files = list(frames_dir.glob("deck.*.png"))
        if not png_files:
            raise Exception(
                f"Marp command succeeded but no PNG frames were generated in {frames_dir}\n"
                f"Expected files like: deck.001.png, deck.002.png, etc.\n"
                f"Directory contents: {list(frames_dir.glob('*'))}"
            )
        
        return str(frames_dir)
```

### Required Libraries
- `marp-cli` (Node.js package installed globally with `npm i -g @marp-team/marp-cli`)

## 2. Generating Audio Files from Slide Content Using Sarvam API

### File Involved
- `processors/audio_generator.py` - Creates audio narration using Sarvam AI TTS API

### Key Code

```python
import os
import json
from pathlib import Path
from sarvamai import SarvamAI
from sarvamai.play import save as sarvam_save

def generate_audio(slides_json_path, output_dir):
    # Verify that the SARVAM_API_KEY is set
    SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY environment variable not set")
    
    # Use the provided output directory
    audio_dir = Path(output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load slides data from JSON
        with open(slides_json_path, 'r', encoding='utf-8') as f:
            slides_data = json.load(f)
        
        # Initialize Sarvam AI client
        client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
        
        # Generate audio for each slide
        for i, slide_info in enumerate(slides_data):
            audio_text = slide_info.get("audio", "")
            # Use 1-based indexing for slide numbers in filenames
            slide_num = slide_info.get("slide number", i + 1)
            
            if not audio_text:
                print(f"  Skipping audio generation for Slide {slide_num}: No narration text provided.")
                continue
            
            print(f"  Generating audio for Slide {slide_num}...")
            audio = client.text_to_speech.convert(
                text=audio_text,
                target_language_code="en-IN",
                model="bulbul:v2",
                speaker="anushka"
            )
            
            # Save audio files as slide01.wav, slide02.wav, etc.
            audio_file_path = audio_dir / f"slide{slide_num:02d}.wav"
            sarvam_save(audio, str(audio_file_path))
            print(f"  Successfully saved audio for Slide {slide_num} to {audio_file_path}")
        
        return str(audio_dir)
        
    except Exception as e:
        raise Exception(f"Error generating audio: {str(e)}")
```

### Required Libraries
- `sarvamai` (Python package installed with `pip install sarvamai`)

## 3. Stitching PNG and Audio Files to Create Video Using ffmpeg

### File Involved
- `processors/video_creator.py` - Assembles the final video using ffmpeg

### Key Code

```python
import os
import sys
import subprocess
import shutil
import time
import tempfile
from pathlib import Path

def create_video(frames_dir, audio_dir, output_path=None, progress_callback=None):
    def update_progress(message, current=None, total=None):
        if progress_callback:
            progress_callback(message, current, total)
        else:
            print(message)
            
    # Determine output video path
    if output_path is None:
        output_dir = Path("slides")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = output_dir / "video.mp4"
    else:
        output_video_path = Path(output_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for ffmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")

    temp_dir = Path(tempfile.mkdtemp(prefix="video_temp_"))
    try:
        # Add a small delay to ensure files are written to disk
        time.sleep(1)

        frames_dir_path = Path(frames_dir)
        audio_dir_path = Path(audio_dir)
        
        if not frames_dir_path.exists():
            raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")

        # Glob for PNG files, which marp-cli names `deck.001.png`, `deck.002.png`, etc.
        png_files = sorted(frames_dir_path.glob("deck.*.png"))
        audio_files = sorted(audio_dir_path.glob("*.wav"))

        if not png_files:
            raise FileNotFoundError(f"No PNG files found in frames directory: {frames_dir}")
        if not audio_files:
            raise FileNotFoundError(f"No audio files found in audio directory: {audio_dir}")

        # Step 1: Pre-process audio files to a standard format to avoid ffmpeg errors
        update_progress("Pre-processing audio files...")
        standardized_audio_files = []
        for i, audio_file in enumerate(audio_files):
            update_progress(f"Processing audio file {i+1}/{len(audio_files)}", i, len(audio_files))
            standardized_path = temp_dir / f"std_{audio_file.name}"
            standardize_cmd = [
                ffmpeg_path,
                "-i", str(audio_file),
                "-acodec", "pcm_s16le", # Standard 16-bit PCM
                "-ar", "44100", # 44.1kHz sample rate
                "-ac", "2", # Stereo
                str(standardized_path)
            ]
            result = subprocess.run(standardize_cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                # Error handling code...
                pass
            standardized_audio_files.append(standardized_path)
        update_progress("Audio pre-processing complete")

        # Step 2: Create individual video clips (image + audio) for each slide
        update_progress("Creating individual video clips...")
        individual_clips = []
        for i, audio_file in enumerate(standardized_audio_files):
            slide_num = i + 1
            update_progress(f"Creating clip {slide_num}/{len(standardized_audio_files)}", i, len(standardized_audio_files))
            
            if len(png_files) <= i:
                update_progress(f"Warning: Missing PNG for slide {slide_num}. Skipping.")
                continue
            png_file = png_files[i]
            
            clip_output_path = temp_dir / f"clip_{slide_num:02d}.mp4"
            
            # Get audio duration from the standardized file
            ffprobe_path = shutil.which("ffprobe") or ffmpeg_path # Use ffprobe if available
            duration_cmd = [ffprobe_path, "-i", str(audio_file), "-show_entries", "format=duration", "-v", "quiet", "-of", "csv=p=0"]
            duration_output = subprocess.check_output(duration_cmd).decode("utf-8").strip()
            duration = float(duration_output)

            # Command to create a video clip for one slide
            clip_cmd = [
                ffmpeg_path, 
                "-loop", "1", 
                "-i", str(png_file),
                "-i", str(audio_file),
                "-c:v", "libx264", 
                "-tune", "stillimage", # Optimize for still images
                "-c:a", "aac", 
                "-b:a", "192k", # Audio bitrate
                "-pix_fmt", "yuv420p", 
                "-shortest", 
                "-t", str(duration), # Set video duration to audio duration
                str(clip_output_path)
            ]
            result = subprocess.run(clip_cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                # Error handling code...
                pass
            individual_clips.append(clip_output_path)
        
        if not individual_clips:
            update_progress("Error: No individual video clips were created")
            return None

        # Step 3: Concatenate individual video clips
        update_progress("Concatenating individual video clips...")
        concat_file_list_path = temp_dir / "concat_clips.txt"
        with open(concat_file_list_path, "w", encoding="utf-8") as f:
            for clip_path in individual_clips:
                f.write(f"file '{clip_path}'\n")
        
        final_concat_cmd = [
            ffmpeg_path, 
            "-f", "concat", 
            "-safe", "0", 
            "-i", str(concat_file_list_path),
            "-c", "copy", 
            str(output_video_path)
        ]
        result = subprocess.run(final_concat_cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            # Error handling code...
            pass
        
        update_progress(f"Video creation completed: {output_video_path}")
        return str(output_video_path)

    except Exception as e:
        update_progress(f"Error creating video: {str(e)}")
        # Re-raise the exception so the caller can handle it and display a detailed error
        raise e
    finally:
        shutil.rmtree(temp_dir)
```

### Required Libraries/Tools
- `ffmpeg` (command-line tool installed on the system)
- `subprocess` (Python standard library for running ffmpeg commands)
- `shutil` (Python standard library for finding ffmpeg executable)