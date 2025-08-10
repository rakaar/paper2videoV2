#!/usr/bin/env python3
"""
Script to generate audio files from presentation.json using Sarvam TTS API.
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI
from sarvamai.play import save as sarvam_save

def load_env():
    """Load environment variables from .env file."""
    load_dotenv()
    return os.getenv("SARVAM_API_KEY")

def generate_audio(text, api_key, output_path):
    """
    Generate audio from text using Sarvam AI TTS.
    
    Args:
        text (str): Text to convert to speech
        api_key (str): Sarvam API key
        output_path (Path): Path to save the audio file
    """
    try:
        # Initialize Sarvam AI client
        client = SarvamAI(api_subscription_key=api_key)
        
        # Generate audio
        audio = client.text_to_speech.convert(
            text=text,
            target_language_code="en-IN",
            model="bulbul:v2",
            speaker="anushka"
        )
        
        # Save the audio file
        sarvam_save(audio, str(output_path))
        
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
