# Audio Generation Issue Analysis

## Project Overview
This project generates audio files from presentation slides using the Sarvam TTS API. The challenge is that Sarvam has a 1500-character limit per request, but some slides have longer text that needs to be split into chunks and then concatenated.

## Files Involved

### Main Files
- **`generate_audio.py`** - Main script to generate audio files from presentation.json
- **`artifacts/test_paper/presentation.json`** - Input file containing slide content including "Audio" text
- **`artifacts/test_paper/audio/slide_XXX.wav`** - Output audio files (one per slide)

### Key Functions
- `_sent_chunks(text)` - Splits text into sentence-aware chunks under 1400 characters
- `generate_audio(text, api_key, output_path)` - Generates audio using Sarvam API and concatenates chunks

## The Problem

### Issue Description
The audio generation for slide 1 is not working correctly:
- Slide 1 has 1661 characters of text, which exceeds Sarvam's 1500-character limit
- The text should be split into 2 chunks and concatenated
- However, the chunking function is treating the entire text as a single sentence
- This results in only one chunk being created instead of two
- The audio is generated for only the first part of the text (truncated)

### Expected vs Actual Behavior
- **Expected**: Slide 1 text (1661 chars) → Split into 2 chunks → Concatenated audio (~20+ seconds)
- **Actual**: Slide 1 text (1661 chars) → Treated as 1 chunk → Truncated audio (~10 seconds)

## Investigation Findings

### 1. Regex Issue
The sentence splitting regex `r'(?<=[.!?])\s+'` should split the text into 12 sentences, but in some contexts it's only producing 1 sentence.

Testing shows:
```python
# This works correctly (12 sentences)
sents = re.split(r'(?<=[.!?])\s+', slide_1_text.strip())
print(len(sents))  # Output: 12

# But in the actual script context, it produces only 1 sentence
```

### 2. Inconsistent Behavior
The same function behaves differently in different contexts:
- When tested directly: Produces 2 chunks correctly
- When called from the main script: Produces 1 chunk incorrectly

### 3. Text Analysis
Slide 1 text analysis:
- Length: 1661 characters
- Periods: 12 periods, all followed by spaces
- Should split into 12 sentences, then 2 chunks (1330 chars + 330 chars)

## What We've Done So Far

### 1. Implemented Chunking Logic
- Created `_sent_chunks()` function to split text at sentence boundaries
- Set MAX_CHARS = 1400 to stay under Sarvam's 1500 limit
- Added logic to concatenate audio chunks using ffmpeg

### 2. Added Debugging
- Added detailed debug output to trace the chunking process
- Verified that the regex should work correctly
- Confirmed that slide 1 text has 12 sentences

### 3. Tested Different Approaches
- Direct function testing: Works correctly
- Script context testing: Fails to split sentences
- Regex pattern testing: All patterns work correctly in isolation

## Why It's Not Working

The root cause appears to be an inconsistency in how the regex splitting behaves in different execution contexts. While the regex `r'(?<=[.!?])\s+'` should split the text into 12 sentences, in the context of the main script it's returning only 1 element.

Possible causes:
1. **Text encoding issues** - The text might be different when loaded in different contexts
2. **Import or environment differences** - Different regex engine behavior
3. **Caching or state issues** - Previous runs affecting current behavior
4. **String manipulation differences** - Text being modified before regex processing

## Next Steps to Fix

1. **Verify text consistency** - Ensure the text loaded in all contexts is identical
2. **Add more robust sentence detection** - Use alternative methods to split sentences
3. **Implement fallback chunking** - If sentence splitting fails, split by character count
4. **Add comprehensive error checking** - Verify each step of the chunking process

## Files to Modify

1. **`generate_audio.py`** - Fix the chunking logic and add better error handling
2. **Add test files** - Create standalone test scripts to verify the fix