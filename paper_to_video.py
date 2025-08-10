import argparse
import asyncio
import logging
import os
from pathlib import Path

# Import the refactored main functions.
# Note: The `summarize_chunks.py` script already had a `main_async` function.
from generate_slides import main_async as generate_main
from plan_slides import main_async as plan_main
from summarize_chunks import main_async as summarize_main


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    ap = argparse.ArgumentParser(description="Full paper-to-video pipeline.")
    ap.add_argument(
        "--pdf-name",
        type=str,
        required=True,
        help="The base name of the paper (e.g., 'test_paper') used to find OCR files and name artifacts.",
    )
    ap.add_argument(
        "--ocr-root",
        type=Path,
        default=Path("mistral_responses"),
        help="Root directory for OCR outputs.",
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
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing for all steps, overwriting existing files.",
    )
    ap.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging for detailed output."
    )
    ap.add_argument(
        "--summarize-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="OpenRouter model slug to use for the summarization step."
    )
    return ap.parse_args()


async def main():
    """Run the end-to-end paper-to-video pipeline."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    pdf_name = args.pdf_name
    ocr_root = args.ocr_root
    artifacts_root = args.artifacts_root

    # Define the directory for this specific paper's artifacts
    paper_artifacts_dir = artifacts_root / pdf_name
    presentation_outdir = args.presentation_outdir or paper_artifacts_dir

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
    )
    try:
        await summarize_main(summarize_args)
        logging.info("--- Step 1: Summarize Chunks finished ---")
    except Exception as e:
        logging.error(f"Step 1 (Summarize) failed: {e}", exc_info=args.verbose)
        return

    # --- Step 2: Plan Slides ---
    logging.info(f"--- Running Step 2: Plan Slides for {pdf_name} ---")
    plan_args = argparse.Namespace(
        summaries_dir=paper_artifacts_dir,
        outdir=paper_artifacts_dir,
        model=os.environ.get("OPENROUTER_PLANNER_MODEL", "deepseek/deepseek-chat"),
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
        model=os.environ.get("OPENROUTER_GENERATOR_MODEL", "openai/gpt-4o"),
        force=args.force,
        verbose=args.verbose,
    )
    try:
        await generate_main(generate_args)
        logging.info("--- Step 3: Generate Slides finished ---")
    except Exception as e:
        logging.error(f"Step 3 (Generate) failed: {e}", exc_info=args.verbose)
        return

    logging.info(
        f"Pipeline finished for {pdf_name}. Final presentation at: {presentation_outdir / 'presentation.json'}"
    )


if __name__ == "__main__":
    asyncio.run(main())
