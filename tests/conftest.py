import os

# keep Inspect's rich UI out of pytest output
os.environ.setdefault("INSPECT_DISPLAY", "none")
os.environ.setdefault("INSPECT_LOG_LEVEL", "warning")
