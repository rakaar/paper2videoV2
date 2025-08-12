import argparse
import asyncio
import logging
import os
import sys
import subprocess
from pathlib import Path

# Import the refactored main functions.
# Note: The `summarize_chunks.py` script already had a `main_async` function.
from generate_slides import main_async as generate_main
from plan_slides import main_async as plan_main
from summarize_chunks import main_async as summarize_main
from make_paper_card import main_async as card_main
from mistral_ocr import run as ocr_run


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    ap = argparse.ArgumentParser(description="Full paper-to-video pipeline.")
    ap.add_argument(
        "--pdf-name",
        type=str,
        required=False,
        default=None,
        help="Base name of the paper (e.g., 'test_paper'). If omitted, inferred from --pdf-file.",
    )
    ap.add_argument(
        "--pdf-file",
        type=Path,
        default=None,
        help="Path to the input PDF to OCR. If provided, OCR will run and outputs will be placed under --ocr-root/<pdf-name>/.",
    )
    ap.add_argument(
        "--ocr-root",
        type=Path,
        default=Path("mistral_responses"),
        help="Root directory for OCR outputs.",
    )
    ap.add_argument(
        "--page-ranges",
        type=str,
        default=None,
        help="OCR page ranges (e.g., '1-3,7-' or '5'). If omitted, OCR processes all pages.",
    )
    ap.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts"),
        help="Root directory for generated artifacts.",
    )
    ap.add_argument(
        "--presentation-outdir",
        type=Path,
        default=None,
        help="Output directory for the final presentation.json. Defaults to <artifacts-root>/<pdf-name>/.",
    )
    # Model selection and generation controls
    ap.add_argument(
        "--summarize-model",
        type=str,
        default=os.environ.get("OPENROUTER_SUMMARIZER_MODEL", "openai/gpt-oss-120b"),
        help="OpenRouter model slug(s) for summarization (comma-separated for fallback).",
    )
    ap.add_argument(
        "--planner-model",
        type=str,
        default=os.environ.get("OPENROUTER_PLANNER_MODEL", "qwen/qwen-2.5-7b-instruct,mistralai/mixtral-8x7b-instruct"),
        help="OpenRouter model slug(s) for slide planning (comma-separated for fallback).",
    )
    ap.add_argument(
        "--planner-max-tokens",
        type=int,
        default=None,
        help="Max tokens for planner responses (omit to uncap).",
    )
    ap.add_argument(
        "--generator-model",
        type=str,
        default=os.environ.get("OPENROUTER_GENERATOR_MODEL", "mistralai/mistral-small-24b-instruct-2501,meta-llama/llama-3.2-3b-instruct"),
        help="OpenRouter model slug(s) for slide generation (comma-separated for fallback).",
    )
    ap.add_argument(
        "--generator-max-tokens",
        type=int,
        default=None,
        help="Max tokens for generator responses (omit to uncap).",
    )
    ap.add_argument(
        "--figure-reuse-limit",
        type=int,
        default=-1,
        help="Max number of times a single figure can be reused across the deck (-1 = unlimited).",
    )
    ap.add_argument(
        "--add-cover",
        action="store_true",
        help="Prepend a cover slide (with narration) using title/authors extracted from OCR via LLM (fallback heuristics).",
    )
    ap.add_argument(
        "--cover-model",
        type=str,
        default=os.environ.get("OPENROUTER_COVER_MODEL", "mistralai/mistral-small-24b-instruct-2501"),
        help="OpenRouter model slug for extracting cover metadata (title/authors).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing for all steps, overwriting existing files.",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for detailed output."
    )
    return ap.parse_args()


async def main():
    """Run the end-to-end paper-to-video pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Determine paper name
    pdf_name = args.pdf_name or (args.pdf_file.stem if args.pdf_file else None)
    if not pdf_name:
        logging.error("You must provide --pdf-name or --pdf-file.")
        return
    ocr_root = args.ocr_root
    artifacts_root = args.artifacts_root

    # Define the directory for this specific paper's artifacts
    paper_artifacts_dir = artifacts_root / pdf_name
    presentation_outdir = args.presentation_outdir or paper_artifacts_dir

    # --- Optional Step 0: OCR ---
    md_dir = ocr_root / pdf_name / "markdown"
    if args.pdf_file:
        try:
            logging.info(f"--- Running OCR: {args.pdf_file} -> {ocr_root}/{pdf_name} ---")
            ocr_run(args.pdf_file, ocr_root, args.page_ranges)
            logging.info("--- OCR finished ---")
        except Exception as e:
            logging.error(f"Step 0 (OCR) failed: {e}", exc_info=args.verbose)
            return
    else:
        if not md_dir.exists():
            logging.error(
                f"OCR pages not found at {md_dir}. Provide --pdf-file to run OCR or ensure OCR outputs exist."
            )
            return

    # --- Step 1: Summarize Chunks ---
    logging.info(f"--- Running Step 1: Summarize Chunks for {pdf_name} ---")
    summarize_args = argparse.Namespace(
        pages_dir=ocr_root / pdf_name / "markdown",
        outdir=paper_artifacts_dir,
        chunk_size=1024,  # Default from summarize_chunks.py
        overlap=128,  # Default from summarize_chunks.py
        force=args.force,
        verbose=args.verbose,
        model=args.summarize_model,
        max_tokens=None,
    )
    try:
        await summarize_main(summarize_args)
        logging.info("--- Step 1: Summarize Chunks finished ---")
    except Exception as e:
        logging.warning(f"Step 1 (Summarize) attempt 1 failed: {e}", exc_info=args.verbose)
        # Retry once: summarize_chunks.py writes partial results before raising.
        try:
            await summarize_main(summarize_args)
            logging.info("--- Step 1: Summarize Chunks finished on retry ---")
        except Exception as e2:
            summaries_path = paper_artifacts_dir / "chunk_summaries.jsonl"
            if summaries_path.exists() and summaries_path.stat().st_size > 0:
                logging.warning(
                    f"Summarization had errors but wrote partial results to {summaries_path}. Continuing with downstream steps."
                )
            else:
                logging.error(f"Step 1 (Summarize) failed after retry: {e2}")
                return

    # --- Step 1.5: Make Paper Card ---
    logging.info(f"--- Running Step 1.5: Make Paper Card for {pdf_name} ---")
    card_args = argparse.Namespace(
        artifacts_dir=paper_artifacts_dir,
        outdir=paper_artifacts_dir,
        model=os.environ.get(
            "OPENROUTER_CARD_MODEL",
            os.environ.get("OPENROUTER_GENERATOR_MODEL", "mistralai/mistral-small-24b-instruct-2501,meta-llama/llama-3.2-3b-instruct"),
        ),
        max_tokens=None,
        verbose=args.verbose,
        force=args.force,
    )
    try:
        await card_main(card_args)
        logging.info("--- Step 1.5: Paper Card finished ---")
    except Exception as e:
        logging.error(f"Step 1.5 (Paper Card) failed: {e}", exc_info=args.verbose)
        return

    # --- Step 2: Plan Slides ---
    logging.info(f"--- Running Step 2: Plan Slides for {pdf_name} ---")
    plan_args = argparse.Namespace(
        summaries_dir=paper_artifacts_dir,
        outdir=paper_artifacts_dir,
        model=args.planner_model,
        max_tokens=args.planner_max_tokens,
        force=args.force,
        verbose=args.verbose,
    )
    try:
        await plan_main(plan_args)
        logging.info("--- Step 2: Plan Slides finished ---")
    except Exception as e:
        logging.error(f"Step 2 (Plan) failed: {e}", exc_info=args.verbose)
        return

    # --- Step 3: Generate Slides ---
    logging.info(f"--- Running Step 3: Generate Slides for {pdf_name} ---")
    generate_args = argparse.Namespace(
        ocr_dir=ocr_root,
        pdf_name=pdf_name,
        artifacts_dir=paper_artifacts_dir,
        outdir=presentation_outdir,
        model=args.generator_model,
        max_tokens=args.generator_max_tokens,
        figure_reuse_limit=args.figure_reuse_limit,
        add_cover=args.add_cover,
        cover_model=args.cover_model,
        force=args.force,
        verbose=args.verbose,
    )
    try:
        await generate_main(generate_args)
        logging.info("--- Step 3: Generate Slides finished ---")
    except Exception as e:
        logging.error(f"Step 3 (Generate) failed: {e}", exc_info=args.verbose)
        return

    presentation_path = presentation_outdir / "presentation.json"
    if not presentation_path.exists():
        logging.error(f"Expected presentation not found at {presentation_path}")
        return

    # --- Step 4: Render PNG slides via Marp ---
    logging.info(f"--- Running Step 4: Render PNGs via Marp for {pdf_name} ---")
    try:
        render_script = Path(__file__).parent / "generate_slide_pngs.py"
        cmd = [
            sys.executable,
            "-u",
            str(render_script),
            "--presentation-file",
            str(presentation_path),
            "--output-dir",
            str(paper_artifacts_dir),
            "--paper-name",
            pdf_name,
            "--ocr-dir",
            str(ocr_root),
        ]
        if args.add_cover:
            # We already added cover during generation; avoid duplicate cover in renderer
            cmd.append("--no-cover")
        logging.debug(f"Render command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logging.info("--- Step 4: Render PNGs finished ---")
    except subprocess.CalledProcessError as e:
        logging.error(f"Step 4 (Render PNGs) failed with exit code {e.returncode}")
        return
    except FileNotFoundError as e:
        logging.error(f"Step 4 (Render PNGs) failed: {e}")
        return

    # --- Step 5: Generate Audio via Sarvam ---
    logging.info(f"--- Running Step 5: Generate Audio for {pdf_name} ---")
    try:
        audio_script = Path(__file__).parent / "generate_audio.py"
        audio_outdir = paper_artifacts_dir / "audio"
        cmd = [
            sys.executable,
            "-u",
            str(audio_script),
            "--presentation-file",
            str(presentation_path),
            "--output-dir",
            str(audio_outdir),
            "--paper-name",
            pdf_name,
        ]
        logging.debug(f"Audio command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logging.info("--- Step 5: Generate Audio finished ---")
    except subprocess.CalledProcessError as e:
        logging.error(f"Step 5 (Generate Audio) failed with exit code {e.returncode}")
        return
    except FileNotFoundError as e:
        logging.error(f"Step 5 (Generate Audio) failed: {e}")
        return

    # --- Step 6: Stitch Video ---
    logging.info(f"--- Running Step 6: Stitch Video for {pdf_name} ---")
    try:
        stitch_script = Path(__file__).parent / "stitch_video.py"
        cmd = [
            sys.executable,
            "-u",
            str(stitch_script),
            "--paper-name",
            pdf_name,
        ]
        logging.debug(f"Stitch command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logging.info("--- Step 6: Stitch Video finished ---")
    except subprocess.CalledProcessError as e:
        logging.error(f"Step 6 (Stitch Video) failed with exit code {e.returncode}")
        return
    except FileNotFoundError as e:
        logging.error(f"Step 6 (Stitch Video) failed: {e}")
        return

    logging.info(f"Pipeline finished for {pdf_name}.")
    logging.info(f"- Presentation: {presentation_path}")
    logging.info(f"- PNGs: {paper_artifacts_dir / 'pngs'}")
    logging.info(f"- Audio: {paper_artifacts_dir / 'audio'}")
    logging.info(f"- Video: {paper_artifacts_dir / 'video.mp4'}")


if __name__ == "__main__":
    asyncio.run(main())
