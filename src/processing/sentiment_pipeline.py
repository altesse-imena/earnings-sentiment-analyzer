"""
FinBERT sentiment pipeline.
Chunks transcripts into sentences, scores each with FinBERT,
and aggregates by speaker role and call section.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TRANSCRIPT_DIR = Path("data/raw/transcripts")
PROCESSED_DIR = Path("data/processed")

MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 512
CHUNK_SIZE = 128  # tokens per chunk for long sentences

# Speaker role patterns
ROLE_PATTERNS = {
    "CEO": re.compile(r"\b(chief executive|ceo|president and ceo)\b", re.I),
    "CFO": re.compile(r"\b(chief financial|cfo|evp finance)\b", re.I),
    "ANALYST": re.compile(r"\b(analyst|research|bank|securities|capital|group)\b", re.I),
    "OPERATOR": re.compile(r"\b(operator|moderator|conference)\b", re.I),
}

# Section boundary patterns
SECTION_PATTERNS = {
    "PREPARED": re.compile(r"\b(prepared remarks|opening remarks|forward.looking)\b", re.I),
    "QA": re.compile(r"\b(question.and.answer|q&a|question and answer|questions)\b", re.I),
}


class FinBERTScorer:
    def __init__(self):
        logger.info(f"Loading FinBERT model: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.labels = ["positive", "negative", "neutral"]
        logger.info(f"FinBERT loaded on {self.device}")

    def score_batch(self, sentences: list[str]) -> list[dict]:
        """Score a batch of sentences. Returns list of {positive, negative, neutral, label}."""
        if not sentences:
            return []

        results = []
        batch_size = 16
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for p in probs:
                scores = {self.labels[j]: float(p[j]) for j in range(3)}
                scores["label"] = self.labels[int(np.argmax(p))]
                scores["sentiment_score"] = scores["positive"] - scores["negative"]
                results.append(scores)

        return results


def split_sentences(text: str) -> list[str]:
    """Split transcript text into sentences, skipping very short lines."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def detect_speaker_role(line: str) -> str:
    """Detect the speaker role from a line of transcript text."""
    for role, pattern in ROLE_PATTERNS.items():
        if pattern.search(line):
            return role
    return "OTHER"


def detect_section(text: str, position: float) -> str:
    """
    Detect call section (PREPARED vs QA) based on keyword presence and position.
    position: relative position in transcript (0.0 to 1.0)
    """
    for section, pattern in SECTION_PATTERNS.items():
        if pattern.search(text):
            return section
    # Heuristic: first 60% of transcript is usually prepared remarks
    return "PREPARED" if position < 0.6 else "QA"


def parse_transcript(text: str) -> list[dict]:
    """
    Parse a raw transcript into a list of sentence records with metadata.
    Each record: {sentence, position, speaker_role, section}
    """
    lines = text.split("\n")
    records = []
    total_lines = len(lines)
    current_speaker = "UNKNOWN"

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Detect speaker role changes (lines that look like speaker introductions)
        if len(line) < 80 and any(p.search(line) for p in ROLE_PATTERNS.values()):
            current_speaker = detect_speaker_role(line)
            continue

        position = i / max(total_lines, 1)
        section = detect_section(line, position)
        sentences = split_sentences(line)

        for sentence in sentences:
            records.append({
                "sentence": sentence,
                "position": round(position, 4),
                "speaker_role": current_speaker,
                "section": section,
            })

    return records


def aggregate_scores(df: pd.DataFrame) -> dict:
    """Aggregate sentence-level scores into call-level features."""
    features = {}

    # Overall scores
    features["overall_sentiment"] = df["sentiment_score"].mean()
    features["sentiment_volatility"] = df["sentiment_score"].std()
    features["positive_ratio"] = (df["label"] == "positive").mean()
    features["negative_ratio"] = (df["label"] == "negative").mean()
    features["neutral_ratio"] = (df["label"] == "neutral").mean()

    # By section
    for section in ["PREPARED", "QA"]:
        sub = df[df["section"] == section]
        if not sub.empty:
            features[f"{section.lower()}_sentiment"] = sub["sentiment_score"].mean()
            features[f"{section.lower()}_volatility"] = sub["sentiment_score"].std()
        else:
            features[f"{section.lower()}_sentiment"] = np.nan
            features[f"{section.lower()}_volatility"] = np.nan

    # Tone shift: QA minus prepared — guard against NaN when a section is absent
    qa_s   = features.get("qa_sentiment", np.nan)
    prep_s = features.get("prepared_sentiment", np.nan)
    features["tone_shift"] = (qa_s - prep_s) if pd.notna(qa_s) and pd.notna(prep_s) else 0.0

    # By speaker role
    for role in ["CEO", "CFO", "ANALYST"]:
        sub = df[df["speaker_role"] == role]
        if not sub.empty:
            features[f"{role.lower()}_sentiment"] = sub["sentiment_score"].mean()
            features[f"{role.lower()}_sentence_count"] = len(sub)
        else:
            features[f"{role.lower()}_sentiment"] = np.nan
            features[f"{role.lower()}_sentence_count"] = 0

    features["total_sentences"] = len(df)
    return features


def process_transcript_file(path: Path, scorer: FinBERTScorer) -> pd.DataFrame | None:
    """Process a single transcript file and return a sentence-level scored DataFrame."""
    logger.info(f"Processing: {path.name}")
    text = path.read_text(errors="ignore")
    if len(text) < 200:
        logger.warning(f"Transcript too short, skipping: {path.name}")
        return None

    records = parse_transcript(text)
    if not records:
        return None

    sentences = [r["sentence"] for r in records]
    scores = scorer.score_batch(sentences)

    for rec, score in zip(records, scores):
        rec.update(score)

    df = pd.DataFrame(records)
    df["file"] = path.name
    return df


def run_pipeline():
    """Process all transcripts and save features to /data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    scorer = FinBERTScorer()

    all_features = []
    sentence_dfs = []

    transcript_files = list(TRANSCRIPT_DIR.rglob("*.txt"))
    if not transcript_files:
        logger.warning(f"No transcript files found in {TRANSCRIPT_DIR}")
        return

    logger.info(f"Found {len(transcript_files)} transcripts to process")

    for path in transcript_files:
        ticker = path.parent.name
        # Parse date from filename (YYYY-MM-DD_...)
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", path.name)
        event_date = date_match.group(1) if date_match else "unknown"

        df = process_transcript_file(path, scorer)
        if df is None:
            continue

        df["ticker"] = ticker
        df["event_date"] = event_date
        sentence_dfs.append(df)

        features = aggregate_scores(df)
        features["ticker"] = ticker
        features["event_date"] = event_date
        features["source_file"] = path.name
        all_features.append(features)

    if sentence_dfs:
        sentences_out = PROCESSED_DIR / "sentences_scored.csv"
        pd.concat(sentence_dfs, ignore_index=True).to_csv(sentences_out, index=False)
        logger.info(f"Saved sentence scores → {sentences_out}")

    if all_features:
        features_out = PROCESSED_DIR / "call_features.csv"
        pd.DataFrame(all_features).to_csv(features_out, index=False)
        logger.info(f"Saved call features → {features_out}")
        logger.info(f"Processed {len(all_features)} transcripts successfully")
    else:
        logger.warning("No features generated — check transcript files")


if __name__ == "__main__":
    run_pipeline()
