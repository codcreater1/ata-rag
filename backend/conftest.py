"""Put the backend directory on sys.path so tests can import ``app`` regardless
of where pytest is invoked from. No fixtures, no app changes — purely for
collection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
