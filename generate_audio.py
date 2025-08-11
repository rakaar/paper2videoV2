#!/usr/bin/env python3
"""
Script to generate audio files from presentation.json using Sarvam TTS API.
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.play import save as sarvam_save

# Character limit for Sarvam TTS (slightly under the 1500 limit for safety)
MAX_CHARS = 1400

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    return os.getenv("SARVAM_API_KEY")

def _sent_chunks(text, max_chars=MAX_CHARS):
    """
    Split text into sentence-aware chunks that don't exceed max_chars.
    
    Args:
        text (str): Text to split into chunks
        max_chars (int): Maximum characters per chunk
        
    Returns:
        list: List of text chunks
    """
    # Split text into sentences
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    out, cur = [], []
    cur_len = 0
    
    for s in sents:
        # Calculate new length if we add this sentence
        new_len = cur_len + len(s) + (1 if cur else 0)
        
        # If adding this sentence would exceed the limit
        if new_len > max_chars:
            # Save current chunk if it exists
            if cur:
                out.append(" ".join(cur))
            # Start new chunk with current sentence
            cur = [s]
            cur_len = len(s)
        else:
            # Add sentence to current chunk
            cur.append(s)
            cur_len = new_len
    
    # Add the last chunk if it exists
    if cur:
        out.append(" ".join(cur))
    
    # Filter out empty chunks
    return [c for c in out if c]

def generate_audio(text, api_key, output_path):
    """
    Generate audio from text using Sarvam AI TTS, handling long texts by chunking.
    
    Args:
        text (str): Text to convert to speech
        api_key (str): Sarvam API key
        output_path (Path): Path to save the audio file
    """
    try:
        # Initialize Sarvam AI client
        client = SarvamAI(api_subscription_key=api_key)
        
        # Split text into chunks if it's too long
        chunks = _sent_chunks(text)
        
        # If no chunks, nothing to generate
        if not chunks:
            print("Warning: No audio content to generate")
            return False
            
        # If only one chunk, generate audio directly
        if len(chunks) == 1:
            audio = client.text_to_speech.convert(
                text=chunks[0],
                target_language_code="en-IN",
                model="bulbul:v2",
                speaker="anushka",
                speech_sample_rate=24000,
                output_audio_codec="wav"
            )
            sarvam_save(audio, str(output_path))
            print(f"Generated audio for slide saved to: {output_path}")
            return True
            
        # For multiple chunks, generate each and concatenate
        with tempfile.TemporaryDirectory() as tmpd:
            part_paths = []
            
            # Generate audio for each chunk
            for idx, chunk in enumerate(chunks, 1):
                audio = client.text_to_speech.convert(
                    text=chunk,
                    target_language_code="en-IN",
                    model="bulbul:v2",
                    speaker="anushka",
                    speech_sample_rate=24000,
                    output_audio_codec="wav"
                )
                
                part_path = Path(tmpd) / f"part_{idx:03d}.wav"
                sarvam_save(audio, str(part_path))
                part_paths.append(part_path)
                
            # Create concat file for ffmpeg
            concat_file = Path(tmpd) / "concat.txt"
            with open(concat_file, 'w') as f:
                for part_path in part_paths:
                    f.write(f"file '{part_path.as_posix()}'\n")
                    
            # Concatenate audio files using ffmpeg
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
                 "-c", "copy", str(output_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
        print(f"Generated audio for slide saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error generating audio for slide: {e}")
        return False

def main():
    """Main function to generate audio files from presentation.json."""
    parser = argparse.ArgumentParser(description="Generate audio files from presentation.json using Sarvam TTS API.")
    parser.add_argument(
        "--presentation-file",
        type=Path,
        required=True,
        help="Path to the presentation.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the generated audio files.",
    )
    parser.add_argument(
        "--paper-name",
        type=str,
        required=True,
        help="Name of the paper (e.g., 'test_paper')."
    )
    args = parser.parse_args()

    presentation_file = args.presentation_file
    output_dir = args.output_dir
    paper_name = args.paper_name

    if not presentation_file.exists():
        print(f"Error: Presentation file not found at {presentation_file}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load API key
    api_key = load_env()
    if not api_key:
        print("Error: SARVAM_API_KEY not found in .env file")
        return

    # Load presentation data
    with open(presentation_file, 'r') as f:
        presentation_data = json.load(f)

    # Generate audio for each slide
    success_count = 0
    total_slides = len(presentation_data["presentation"])
    
    for i, slide in enumerate(presentation_data["presentation"]):
        slide_number = i + 1
        audio_text = slide.get("Audio")
        
        if not audio_text:
            print(f"Warning: No audio content found for slide {slide_number}")
            continue
            
        # Define output file path
        output_file = output_dir / f"slide_{slide_number:03d}.wav"
        
        # Generate audio
        if generate_audio(audio_text, api_key, output_file):
            success_count += 1
        else:
            print(f"Failed to generate audio for slide {slide_number}")
    
    print(f"\nAudio generation complete: {success_count}/{total_slides} slides successful")

if __name__ == "__main__":
    main()
