"""Put the repo root on sys.path so `import bioreef` resolves under pytest."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
