#!/usr/bin/env python3
"""Grammar scoring engine for spoken audio (SHL Hiring Assessment 2026)."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import whisper
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RobertaModel, RobertaTokenizerFast

try:
    import language_tool_python
except Exception:  # pragma: no cover - optional dependency/runtime
    language_tool_python = None


FILLERS = {
    "uh",
    "um",
    "er",
    "ah",
    "hmm",
    "like",
    "you know",
    "i mean",
    "sort of",
    "kind of",
}


@dataclass
class ASROutput:
    transcript: str
    duration_sec: float


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _sentence_split(text: str) -> List[str]:
    candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return candidates if candidates else ([text.strip()] if text.strip() else [])


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


class GrammarScoringEngine:
    def __init__(self, whisper_model_name: str = "base") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(whisper_model_name, device=self.device)

        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.gpt2_model.eval()

        self.roberta_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.roberta_model = RobertaModel.from_pretrained("roberta-base").to(self.device)
        self.roberta_model.eval()

        self.grammar_tool = None
        if language_tool_python is not None and shutil.which("java"):
            try:
                self.grammar_tool = language_tool_python.LanguageTool("en-US")
            except Exception:
                self.grammar_tool = None

    def _prepare_audio(self, wav_path: Path) -> Tuple[Path, float]:
        y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        duration = librosa.get_duration(y=y, sr=16000)

        if y.size:
            y, _ = librosa.effects.trim(y, top_db=30)
            peak = np.max(np.abs(y))
            if peak > 0:
                y = y / peak

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()
        sf.write(str(temp_path), y, 16000)
        return temp_path, float(duration)

    def transcribe_audio(self, wav_path: Path) -> ASROutput:
        temp_path, duration = self._prepare_audio(wav_path)
        try:
            result = self.whisper_model.transcribe(
                str(temp_path), language="en", fp16=(self.device == "cuda")
            )
            transcript = (result.get("text") or "").strip()
            return ASROutput(transcript=transcript, duration_sec=duration)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _grammar_error_features(self, text: str) -> Dict[str, float]:
        features: Dict[str, float] = {
            "grammar_errors_total": 0.0,
            "grammar_errors_per_100_words": 0.0,
        }
        words = _word_tokens(text)

        if not self.grammar_tool or not text.strip():
            return features

        matches = self.grammar_tool.check(text)
        features["grammar_errors_total"] = float(len(matches))
        features["grammar_errors_per_100_words"] = _safe_div(len(matches) * 100.0, len(words))

        per_category: Dict[str, int] = {}
        for m in matches:
            category = "unknown"
            if getattr(m, "category", None) and getattr(m.category, "id", None):
                category = str(m.category.id).lower()
            elif getattr(m, "ruleIssueType", None):
                category = str(m.ruleIssueType).lower()
            category = re.sub(r"[^a-z0-9]+", "_", category).strip("_") or "unknown"
            per_category[category] = per_category.get(category, 0) + 1

        for category, count in per_category.items():
            features[f"lt_count_{category}"] = float(count)
            features[f"lt_rate_{category}"] = _safe_div(count * 100.0, len(words))

        return features

    def _fluency_structure_features(self, text: str, duration_sec: float) -> Dict[str, float]:
        words = _word_tokens(text)
        sentences = _sentence_split(text)

        fillers_count = 0
        lowered = text.lower()
        for filler in FILLERS:
            fillers_count += lowered.count(filler)

        repetition_count = 0
        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                repetition_count += 1

        incomplete_proxy = 0
        for sentence in sentences:
            if sentence and sentence[-1] not in ".!?":
                incomplete_proxy += 1

        return {
            "duration_sec": float(duration_sec),
            "word_count": float(len(words)),
            "sentence_count": float(len(sentences)),
            "words_per_sec": _safe_div(len(words), duration_sec),
            "mean_sentence_len": _safe_div(len(words), len(sentences)),
            "repetition_count": float(repetition_count),
            "repetition_rate": _safe_div(repetition_count, len(words)),
            "filler_count": float(fillers_count),
            "filler_rate": _safe_div(fillers_count, len(words)),
            "type_token_ratio": _safe_div(len(set(words)), len(words)),
            "incomplete_sentence_proxy": float(incomplete_proxy),
        }

    @torch.inference_mode()
    def _sentence_perplexity(self, sentence: str) -> float:
        sentence = sentence.strip()
        if not sentence:
            return 0.0

        enc = self.gpt2_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.gpt2_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss.item()
        return float(math.exp(loss))

    def _lm_features(self, text: str) -> Dict[str, float]:
        sentences = _sentence_split(text)
        if not sentences:
            return {"ppl_mean": 0.0, "ppl_max": 0.0, "ppl_std": 0.0}

        ppls = np.array([self._sentence_perplexity(s) for s in sentences], dtype=np.float32)
        return {
            "ppl_mean": float(np.mean(ppls)),
            "ppl_max": float(np.max(ppls)),
            "ppl_std": float(np.std(ppls)),
        }

    @torch.inference_mode()
    def _roberta_embedding(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(768, dtype=np.float32)

        enc = self.roberta_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [1, T, H]
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled.squeeze(0).cpu().numpy().astype(np.float32)

    def build_features(self, transcripts: Sequence[str], durations: Sequence[float]) -> Tuple[pd.DataFrame, np.ndarray]:
        rows: List[Dict[str, float]] = []
        embeddings: List[np.ndarray] = []

        for text, duration in tqdm(list(zip(transcripts, durations)), desc="Extracting text features"):
            feats = {}
            feats.update(self._grammar_error_features(text))
            feats.update(self._fluency_structure_features(text, duration))
            feats.update(self._lm_features(text))
            rows.append(feats)
            embeddings.append(self._roberta_embedding(text))

        tabular = pd.DataFrame(rows).fillna(0.0)
        embedding_matrix = np.vstack(embeddings).astype(np.float32)
        return tabular, embedding_matrix


def detect_columns(df: pd.DataFrame, require_target: bool = True) -> Tuple[str, Optional[str]]:
    audio_candidates = ["audio_path", "audio", "path", "file", "filename"]
    target_candidates = ["score", "mos", "grammar_score", "label", "target"]

    audio_col = next((c for c in audio_candidates if c in df.columns), None)
    if not audio_col:
        for col in df.columns:
            if df[col].astype(str).str.endswith(".wav").any():
                audio_col = col
                break

    if not audio_col:
        raise ValueError("Could not detect audio path column. Expected one of audio_path/audio/path/file/filename.")

    target_col = next((c for c in target_candidates if c in df.columns), None)
    if require_target and not target_col:
        raise ValueError("Could not detect target score column. Expected one of score/mos/grammar_score/label/target.")

    return audio_col, target_col


def _resolve_audio_path(audio_root: Path, raw_value: str) -> Path:
    p = Path(str(raw_value))
    return p if p.is_absolute() else (audio_root / p)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_training(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv) if args.test_csv else None

    if args.max_samples:
        train_df = train_df.head(args.max_samples).copy()
        if test_df is not None:
            test_df = test_df.head(args.max_samples).copy()

    audio_root = Path(args.audio_root).resolve() if args.audio_root else Path(args.train_csv).resolve().parent

    train_audio_col, target_col = detect_columns(train_df, require_target=True)
    train_df = train_df.copy()

    engine = GrammarScoringEngine(whisper_model_name=args.whisper_model)

    transcripts_train: List[str] = []
    durations_train: List[float] = []

    print("Transcribing training audio...")
    for raw_path in tqdm(train_df[train_audio_col].astype(str).tolist(), desc="ASR(train)"):
        wav_path = _resolve_audio_path(audio_root, raw_path)
        asr_out = engine.transcribe_audio(wav_path)
        transcripts_train.append(asr_out.transcript)
        durations_train.append(asr_out.duration_sec)

    train_df["transcript"] = transcripts_train
    tab_train, emb_train = engine.build_features(transcripts_train, durations_train)

    if test_df is not None:
        test_audio_col, _ = detect_columns(test_df, require_target=False)
        test_df = test_df.copy()

        transcripts_test: List[str] = []
        durations_test: List[float] = []

        print("Transcribing test audio...")
        for raw_path in tqdm(test_df[test_audio_col].astype(str).tolist(), desc="ASR(test)"):
            wav_path = _resolve_audio_path(audio_root, raw_path)
            asr_out = engine.transcribe_audio(wav_path)
            transcripts_test.append(asr_out.transcript)
            durations_test.append(asr_out.duration_sec)

        test_df["transcript"] = transcripts_test
        tab_test, emb_test = engine.build_features(transcripts_test, durations_test)

        tab_all = pd.concat([tab_train, tab_test], axis=0, ignore_index=True).fillna(0.0)
        tab_train = tab_all.iloc[: len(tab_train)].reset_index(drop=True)
        tab_test = tab_all.iloc[len(tab_train) :].reset_index(drop=True)
    else:
        tab_train = tab_train.fillna(0.0)
        tab_test = None
        emb_test = None

    X_train = np.hstack([tab_train.to_numpy(dtype=np.float32), emb_train])
    y_train = train_df[target_col].to_numpy(dtype=np.float32)

    models = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=2.0))]),
        "svr": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(C=8.0, epsilon=0.15, kernel="rbf", gamma="scale")),
            ]
        ),
        "hgb": HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=500,
            random_state=args.seed,
            l2_regularization=0.01,
        ),
    }

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_by_model = {name: np.zeros(len(train_df), dtype=np.float32) for name in models}

    print("Running 5-fold CV...")
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_train), start=1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y_train[tr_idx]

        for model_name, model in models.items():
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va).astype(np.float32)
            oof_by_model[model_name][va_idx] = preds

        print(f"Completed fold {fold_idx}/{args.folds}")

    oof_blend = np.mean(np.column_stack([oof_by_model[n] for n in models]), axis=1)
    oof_blend = np.clip(oof_blend, 0.0, 5.0)

    cv_rmse = _rmse(y_train, oof_blend)
    cv_pearson = float(pearsonr(y_train, oof_blend)[0]) if len(y_train) > 1 else 0.0

    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

    train_preds = np.mean(
        np.column_stack([fitted_models[name].predict(X_train).astype(np.float32) for name in fitted_models]),
        axis=1,
    )
    train_preds = np.clip(train_preds, 0.0, 5.0)

    train_rmse = _rmse(y_train, train_preds)
    train_pearson = float(pearsonr(y_train, train_preds)[0]) if len(y_train) > 1 else 0.0

    print("\nSection 7 — Training & OOF Metrics")
    print(f"Training RMSE  : {train_rmse:.6f}")
    print(f"Training Pearson: {train_pearson:.6f}")
    print(f"5-fold OOF RMSE : {cv_rmse:.6f}")
    print(f"5-fold OOF Pearson: {cv_pearson:.6f}")

    metrics = {
        "train_rmse": train_rmse,
        "train_pearson": train_pearson,
        "cv_rmse": cv_rmse,
        "cv_pearson": cv_pearson,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)) if test_df is not None else 0,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    oof_df = pd.DataFrame(
        {
            "y_true": y_train,
            "oof_ridge": oof_by_model["ridge"],
            "oof_svr": oof_by_model["svr"],
            "oof_hgb": oof_by_model["hgb"],
            "oof_blend": oof_blend,
        }
    )
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    train_df.to_csv(out_dir / "train_with_transcripts.csv", index=False)

    if test_df is not None and tab_test is not None and emb_test is not None:
        X_test = np.hstack([tab_test.to_numpy(dtype=np.float32), emb_test])
        test_preds = np.mean(
            np.column_stack([fitted_models[name].predict(X_test).astype(np.float32) for name in fitted_models]),
            axis=1,
        )
        test_preds = np.clip(test_preds, 0.0, 5.0)

        pred_df = test_df.copy()
        pred_df["predicted_score"] = test_preds
        pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
        test_df.to_csv(out_dir / "test_with_transcripts.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grammar scoring engine for spoken audio")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV containing audio paths and scores")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional path to test CSV containing audio paths")
    parser.add_argument("--audio_root", type=str, default=None, help="Root directory for relative audio paths")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for artifacts and predictions")
    parser.add_argument("--whisper_model", type=str, default="base", help="Whisper model name")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit for quick smoke runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
