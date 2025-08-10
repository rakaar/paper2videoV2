# Test Paper Artifacts

This directory contains the generated artifacts for the test_paper.

## Contents

- `pngs/` - Generated slide images in PNG format
- `audio/` - Audio files for each slide in WAV format
- `deck.md` - Marp-compatible markdown file used to generate the slides

## Process

The slide images were generated using the `generate_slide_pngs.py` script:

```bash
python3 generate_slide_pngs.py \
  --presentation-file ../presentation.json \
  --output-dir . \
  --paper-name test_paper
```

The audio files were generated using the `generate_audio.py` script:

```bash
python3 generate_audio.py \
  --presentation-file ../presentation.json \
  --output-dir ./audio \
  --paper-name test_paper
```

## Notes

- The images in the slides are properly referenced and should display correctly
- Mathematical expressions are rendered using KaTeX
- The slide dimensions are 16:9 aspect ratio
- Image paths have been corrected to properly reference the source images
- Images are sized to fit within the slides with CSS styling to prevent cutoff
- Audio files are in WAV format, generated using the Sarvam TTS API

## Script Information

The `generate_slide_pngs.py` script is responsible for converting the `presentation.json` file into slide images. It:

1. Reads the presentation data from `presentation.json`
2. Generates a Marp-compatible markdown file (`deck.md`) with proper formatting
3. Uses Marp CLI to convert the markdown into PNG images
4. Organizes the output files into appropriate directories

The script handles various aspects of slide generation:
- Proper sizing of images to fit within slides
- Correct referencing of image files
- Mathematical expression rendering with KaTeX
- Consistent styling across all slides

The `generate_audio.py` script is responsible for converting the audio narration text in `presentation.json` into audio files using the Sarvam TTS API. It:
1. Reads the presentation data from `presentation.json`
2. Extracts the audio narration text for each slide
3. Uses the Sarvam TTS API to convert text to speech
4. Saves the generated audio files in WAV format