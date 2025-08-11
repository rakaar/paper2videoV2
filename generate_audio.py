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
    Split text into sentence-aware chunks.
    This version treats each sentence as a separate chunk to avoid API issues.
    """
    # 1. Normalize whitespace and clean up text
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return []

    print(f"Debug: Chunking text of {len(text)} characters (sentence-per-chunk strategy)")

    # 2. Attempt to split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print(f"Debug: Split into {len(sentences)} sentences.")

    # 3. Validate sentence length and split if necessary
    final_chunks = []
    for sentence in sentences:
        if len(sentence) > max_chars:
            print(f"Debug: Sentence is too long ({len(sentence)} chars), splitting by words.")
            words = sentence.split()
            word_chunk = []
            word_chunk_len = 0
            for word in words:
                word_len = len(word) + (1 if word_chunk else 0)
                if word_chunk_len + word_len > max_chars:
                    final_chunks.append(' '.join(word_chunk))
                    word_chunk = [word]
                    word_chunk_len = len(word)
                else:
                    word_chunk.append(word)
                    word_chunk_len += word_len
            if word_chunk:
                final_chunks.append(' '.join(word_chunk))
        else:
            final_chunks.append(sentence)
            
    print(f"Debug: Final result: {len(final_chunks)} chunks (one per sentence).")
    return final_chunks

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
        print(f"Generating audio for text: {len(text)} characters")
        chunks = _sent_chunks(text)
        print(f"Text split into {len(chunks)} chunks")
        
        # If no chunks, nothing to generate
        if not chunks:
            print("Warning: No audio content to generate")
            return False
            
        # If only one chunk, generate audio directly
        if len(chunks) == 1:
            print("\n--- Text Chunk Sent to Sarvam API ---")
            print(f"Chunk 1/1 ({len(chunks[0])} chars):\n{chunks[0]}\n")
            print("-------------------------------------\n")
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
            print("\n--- Text Chunks Sent to Sarvam API ---")
            for idx, chunk in enumerate(chunks, 1):
                print(f"Chunk {idx}/{len(chunks)} ({len(chunk)} chars):\n{chunk}\n")
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
            print("-------------------------------------\n")
            
            # Create concat file for ffmpeg
            concat_file = Path(tmpd) / "concat.txt"
            with open(concat_file, 'w') as f:
                for part_path in part_paths:
                    f.write(f"file '{part_path.as_posix()}'\n")
                    
            # Concatenate audio files using ffmpeg
            print("--- Running ffmpeg to concatenate audio ---")
            ffmpeg_command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)]
            print(f"Executing command: {' '.join(ffmpeg_command)}")
            subprocess.run(
                ffmpeg_command,
                check=True
            )
            print("-------------------------------------------\n")
            
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
