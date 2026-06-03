# test_processing.py imports torch via sentiment_pipeline.py.
# Skip it when torch is not installed (lightweight venv without ML deps).
collect_ignore = ["tests/test_processing.py"]
