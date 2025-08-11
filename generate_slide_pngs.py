import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import httpx
from dotenv import load_dotenv

def create_marp_markdown(slide_data, paper_name, first: bool = False):
    """Generates Marp-compatible markdown for a single slide.
    If first=True, do not prepend '---' (first slide follows front-matter).
    """
    lines = [] if first else ["---"]
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

def llm_extract_cover(ocr_dir: Path, paper_name: str, model: str, timeout: float = 30.0) -> Tuple[Optional[str], Optional[str]]:
    """Use OpenRouter to extract title/authors from OCR of page 01.
    Falls back to (None, None) on error.
    """
    page_md = ocr_dir / paper_name / "markdown" / f"{paper_name}_page_01.md"
    if not page_md.exists():
        return None, None
    try:
        text = page_md.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None

    # Trim to reduce tokens
    snippet = text[:4000]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None, None
    try:
        with httpx.Client(timeout=timeout) as client:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You extract metadata from the first page of an academic paper."
                            " Return strictly a JSON object with keys: title (string), authors (string)."
                            " If unsure, leave the value empty string."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "First page OCR Markdown follows. Extract title and authors.\n\n" + snippet
                        ),
                    },
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 120,
                "temperature": 0.0,
            }
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            try:
                obj = json.loads(content)
            except Exception:
                return None, None
            title = (obj.get("title") or "").strip() or None
            authors = (obj.get("authors") or "").strip() or None
            return title, authors
    except Exception:
        return None, None

def extract_title_authors(ocr_dir: Path, paper_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract title and authors from OCR page 01 markdown.
    Returns (title, authors) or (None, None) if not found.
    """
    page_md = ocr_dir / paper_name / "markdown" / f"{paper_name}_page_01.md"
    if not page_md.exists():
        return None, None
    try:
        text = page_md.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, None
    lines = text.splitlines()
    title = None
    authors = None
    # Find first markdown heading
    for idx, line in enumerate(lines[:200]):
        if line.lstrip().startswith("#"):
            title = re.sub(r"^#+\\s*", "", line).strip()
            # Next non-empty non-heading line as authors
            for j in range(idx + 1, min(idx + 10, len(lines))):
                cand = lines[j].strip()
                if not cand:
                    continue
                if cand.startswith("#"):
                    break
                # heuristics: likely an author line, avoid FIG / department long lines
                if len(cand) <= 120 and not cand.lower().startswith(("fig.", "figure ", "department")):
                    authors = cand
                    break
            break
    return title, authors

def _sanitize_heading(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    return re.sub(r"^#+\s*", "", text).strip()

def create_cover_markdown(title: str | None, authors: str | None, first: bool = False):
    """Generate cover slide markdown.
    If first=True, do not prepend '---'.
    """
    lines = [] if first else ["---"]
    title = _sanitize_heading(title)
    if title:
        lines.append(f"# {title}")
    else:
        lines.append("# Presentation")
    lines.append("")
    if authors:
        lines.append(f"_{authors}_")
        lines.append("")
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
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=Path("mistral_responses"),
        help="Root directory of OCR outputs (default: mistral_responses)",
    )
    parser.add_argument(
        "--no-cover",
        action="store_true",
        help="Do not prepend a cover slide with title and authors extracted from OCR.",
    )
    parser.add_argument(
        "--cover-model",
        type=str,
        default=os.environ.get("OPENROUTER_COVER_MODEL", "mistralai/mistral-small-24b-instruct-2501"),
        help="OpenRouter model slug to extract cover metadata (default: mistral-small-24b-instruct-2501)",
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

    # Optional cover slide
    if not args.no_cover:
        load_dotenv()  # ensure OPENROUTER_API_KEY is available
        # Try LLM first, then heuristic
        title, authors = llm_extract_cover(args.ocr_dir, paper_name, args.cover_model)
        if not title and not authors:
            title, authors = extract_title_authors(args.ocr_dir, paper_name)
        # If OCR title not found, fall back to the first slide title
        if title is None and presentation_data.get("presentation"):
            title = presentation_data["presentation"][0].get("Title")
        cover_md = create_cover_markdown(title, authors, first=True)
        all_slides_md.append(cover_md)
    for i, slide in enumerate(presentation_data["presentation"]):
        md = create_marp_markdown(slide, paper_name, first=(args.no_cover and i == 0))
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
            ["marp", "--images", "png", "--image-scale", "2", "--allow-local-files", deck_md_path.name],
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