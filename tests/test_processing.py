"""Unit tests for NLP processing module."""

import numpy as np
import pandas as pd
import pytest

from src.processing.sentiment_pipeline import (
    aggregate_scores,
    detect_section,
    detect_speaker_role,
    parse_transcript,
    split_sentences,
)


class TestSplitSentences:
    def test_splits_on_period(self):
        text = "Revenue increased. Margins expanded. Guidance raised."
        sentences = split_sentences(text)
        assert len(sentences) == 3

    def test_filters_short_lines(self):
        text = "Hi. We are pleased to report record revenue growth this quarter."
        sentences = split_sentences(text)
        assert all(len(s) > 20 for s in sentences)

    def test_empty_string_returns_empty(self):
        assert split_sentences("") == []


class TestDetectSpeakerRole:
    def test_detects_ceo(self):
        assert detect_speaker_role("John Smith, Chief Executive Officer") == "CEO"

    def test_detects_cfo(self):
        assert detect_speaker_role("Jane Doe, Chief Financial Officer") == "CFO"

    def test_detects_analyst(self):
        assert detect_speaker_role("Mike Johnson, Goldman Sachs analyst") == "ANALYST"

    def test_unknown_role(self):
        assert detect_speaker_role("This is a regular sentence.") == "OTHER"


class TestDetectSection:
    def test_prepared_remarks_keyword(self):
        assert detect_section("These are our prepared remarks", 0.1) == "PREPARED"

    def test_qa_keyword(self):
        assert detect_section("We will now open for question and answer", 0.7) == "QA"

    def test_position_heuristic_early(self):
        assert detect_section("Some generic text", 0.3) == "PREPARED"

    def test_position_heuristic_late(self):
        assert detect_section("Some generic text", 0.8) == "QA"


class TestAggregateScores:
    def _make_df(self):
        return pd.DataFrame({
            "sentence": ["Rev grew.", "Margins up.", "Outlook?", "Q1 strong.", "Concerns?"],
            "sentiment_score": [0.5, 0.4, -0.1, 0.3, -0.2],
            "label": ["positive", "positive", "neutral", "positive", "negative"],
            "section": ["PREPARED", "PREPARED", "QA", "PREPARED", "QA"],
            "speaker_role": ["CEO", "CFO", "ANALYST", "CEO", "ANALYST"],
            "positive": [0.7, 0.6, 0.3, 0.6, 0.2],
            "negative": [0.2, 0.2, 0.4, 0.3, 0.4],
            "neutral": [0.1, 0.2, 0.3, 0.1, 0.4],
        })

    def test_overall_sentiment_computed(self):
        df = self._make_df()
        features = aggregate_scores(df)
        assert "overall_sentiment" in features
        assert abs(features["overall_sentiment"] - df["sentiment_score"].mean()) < 1e-6

    def test_tone_shift_computed(self):
        df = self._make_df()
        features = aggregate_scores(df)
        assert "tone_shift" in features
        expected = features["qa_sentiment"] - features["prepared_sentiment"]
        assert abs(features["tone_shift"] - expected) < 1e-6

    def test_positive_negative_ratios_sum_to_at_most_one(self):
        df = self._make_df()
        features = aggregate_scores(df)
        assert features["positive_ratio"] + features["negative_ratio"] <= 1.0 + 1e-9

    def test_section_sentiments_present(self):
        df = self._make_df()
        features = aggregate_scores(df)
        assert "prepared_sentiment" in features
        assert "qa_sentiment" in features

    def test_speaker_sentiments_present(self):
        df = self._make_df()
        features = aggregate_scores(df)
        assert "ceo_sentiment" in features
        assert "cfo_sentiment" in features
        assert "analyst_sentiment" in features

    def test_empty_section_gives_nan(self):
        df = self._make_df()
        df["section"] = "PREPARED"  # no QA rows
        features = aggregate_scores(df)
        assert np.isnan(features["qa_sentiment"])


class TestParseTranscript:
    def test_returns_list_of_dicts(self):
        text = "Good morning. We are pleased to report strong results this quarter.\nRevenue grew 15% year over year."
        records = parse_transcript(text)
        assert isinstance(records, list)
        if records:
            assert "sentence" in records[0]
            assert "speaker_role" in records[0]
            assert "section" in records[0]

    def test_empty_text_returns_empty_list(self):
        assert parse_transcript("") == []
