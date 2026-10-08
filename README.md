# SHL-Hiring-Assessment-2026

Grammar scoring engine for **SHL Hiring Assessment 2026**.

## Task
Predict a continuous MOS-Likert grammar score (0–5) from a 45–60s `.wav` clip.

## Pipeline
`.wav` → mono 16 kHz + silence trim + peak normalize → Whisper ASR transcript →
feature extraction (`LanguageTool` grammar errors, fluency/structure stats, GPT-2 perplexity, RoBERTa embedding) →
Ridge + SVR + HistGradientBoosting regressors → averaged prediction clipped to `[0, 5]`.

## What is implemented
- Whisper-based transcription keeping disfluency-rich text signal.
- Feature blocks:
  - Grammar-error counts/rates from LanguageTool (when Java is available).
  - Fluency/structure features (words/sec, sentence stats, fillers, repetitions, type-token ratio, incomplete sentence proxy).
  - GPT-2 sentence perplexity features (mean/max/std).
  - Mean-pooled RoBERTa semantic/syntactic embeddings.
- 5-fold CV training with three regressors (Ridge, SVR, HistGradientBoosting) and blend averaging.
- Section 7 metric reporting:
  - Training RMSE/Pearson.
  - OOF (5-fold CV) RMSE/Pearson.
- Optional test-set prediction export.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> LanguageTool features are enabled automatically only if Java is available in the environment.

## Usage
```bash
python grammar_scoring_engine.py \
  --train_csv /absolute/path/to/train.csv \
  --test_csv /absolute/path/to/test.csv \
  --audio_root /absolute/path/to/audio_root \
  --output_dir /absolute/path/to/outputs
```

Expected columns:
- Train CSV: one audio-path column (`audio_path`/`audio`/`path`/`file`/`filename`) and one target column (`score`/`mos`/`grammar_score`/`label`/`target`).
- Test CSV: one audio-path column.

## Output artifacts
Written under `--output_dir`:
- `metrics.json`
- `oof_predictions.csv`
- `train_with_transcripts.csv`
- `test_predictions.csv` (if test CSV provided)
- `test_with_transcripts.csv` (if test CSV provided)

## Notes on evaluation
- Training RMSE is optimistic (fit and evaluated on the same labels).
- 5-fold OOF RMSE/Pearson is the reliable estimate of generalization.
- No external training data is used; only provided competition data is intended for fitting.
