"""Test bootstrap: use synthetic fixtures, never a real account snapshot."""
from pathlib import Path
from fundkit import adapters

adapters.SNAPSHOT_FILE = str(Path(__file__).parent / "fixtures" / "account_snapshot.json")