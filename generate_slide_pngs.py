import argparse
import json
import os
import subprocess
from pathlib import Path

def create_marp_markdown(slide_data, paper_name):
    """Generates Marp-compatible markdown for a single slide."""
    lines = ["---"]
    lines.append(f"# {slide_data['Title']}")
    lines.append("")
    lines.append(slide_data['Content'])
    lines.append("")

    if slide_data.get("Figures"):
        for fig in slide_data["Figures"]:
            # Handle different figure formats
            if 'path' in fig:
                img_path = Path(fig['path'])
                # deck.md is in artifacts/test_paper/
                # images are in mistral_responses/test_paper/images
                # so the relative path is ../../mistral_responses/test_paper/images/
                base_path = Path("../../mistral_responses") / paper_name / "images"
                
                # Extract the image number from the path
                # e.g., "images/test_paper_page_01_img-0.jpeg" -> "img-0.jpeg"
                img_name = img_path.name
                if "img-" in img_name:
                    # Extract the part after the last underscore or slash
                    parts = img_name.split("_")
                    if len(parts) > 1:
                        actual_img_name = parts[-1]  # Get "img-0.jpeg"
                    else:
                        actual_img_name = img_name
                else:
                    actual_img_name = img_name
                
                actual_img_path = base_path / actual_img_name
                
                lines.append(f"![{fig.get('caption', '')}]({actual_img_path})")
                lines.append("")
            # Skip figures that don't have a 'path' key (like the one with FigureID and Caption)

    return "\n".join(lines)

def main():
    """Main function to generate PNGs from presentation.json."""
    parser = argparse.ArgumentParser(description="Generate PNG slides from a presentation.json file.")
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
        help="Directory to save the generated markdown, PNGs, and audio.",
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

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    png_output_dir = output_dir / "pngs"
    audio_output_dir = output_dir / "audio"
    os.makedirs(png_output_dir, exist_ok=True)
    os.makedirs(audio_output_dir, exist_ok=True)

    with open(presentation_file, 'r') as f:
        presentation_data = json.load(f)

    all_slides_md = []
    for i, slide in enumerate(presentation_data["presentation"]):
        md = create_marp_markdown(slide, paper_name)
        all_slides_md.append(md)

        # Placeholder for TTS generation
        audio_file_path = audio_output_dir / f"slide_{i+1:03d}.mp3"
        narration = slide.get("Audio")
        if narration:
            print(f"TODO: Generate TTS for slide {i+1} and save to {audio_file_path}")
            # Here you would call the Sarvam AI TTS API
            # For example:
            # sarvam_tts(narration, audio_file_path)

    deck_md_path = output_dir / "deck.md"
    with open(deck_md_path, 'w') as f:
        f.write("""---
marp: true
math: katex
paginate: false
size: 16:9
theme: default
style: |
  section {
    font-size: 18px;
    padding: 20px;
  }
  h1 {
    font-size: 32px;
    text-align: center;
  }
  h2 {
    font-size: 28px;
  }
  img {
    display: block;
    margin: 10px auto;
    max-width: 90%;
    max-height: 300px;
    object-fit: contain;
  }
  .center {
    text-align: center;
  }
---

""")
        f.write('\n'.join(all_slides_md))

    print(f"Generated markdown deck at: {deck_md_path}")

    # Generate PNGs using marp
    try:
        print("Generating PNGs using marp...")
        # Using --image-scale 2 for higher resolution images
        # The output files will be named `deck.001.png`, `deck.002.png`, etc.
        # We need to move them to the pngs directory.
        result = subprocess.run(
            ["marp", "--images", "png", "--image-scale", "2", "--allow-local-files", str(deck_md_path)],
            cwd=output_dir,
            capture_output=True,
            text=True
        )
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"Error running marp: exit code {result.returncode}")
            return

        # Move generated PNGs to the pngs directory
        for item in os.listdir(output_dir):
            if item.startswith("deck.") and item.endswith(".png"):
                os.rename(output_dir / item, png_output_dir / item)

        print(f"Generated PNGs in: {png_output_dir}")

    except FileNotFoundError:
        print("Error: 'marp' command not found. Please ensure Marp CLI is installed and in your PATH.")
        print("Installation instructions: https://github.com/marp-team/marp-cli")
    except subprocess.CalledProcessError as e:
        print(f"Error running marp: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")


if __name__ == "__main__":
    main()