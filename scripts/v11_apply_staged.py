"""Install the pinned B6 v1.1 source bundle; no production trading.

GitHub API transport split the tar.xz into four binary parts.
The SHA256 check ensures the downloaded source is bit-for-bit identical.
"""
from pathlib import Path
import hashlib
import shutil
import sys
import tarfile
import tempfile
import subprocess

BASE = Path(__file__).resolve().parent.parent
PARTS = [BASE / "v11_upload" / f"part-{i:02d}.tar.xz" for i in range(4)]
SHA256 = "9228c6fb0f350399f2cd20104f45e879ee3ea8bc2d672653b325bdcf50d1b05d"

def main():
    payload = b"".join(p.read_bytes() for p in PARTS)
    assert hashlib.sha256(payload).hexdigest() == SHA256, "B6_ARCHIVE_CHECKSUM_MISMATCH"
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        archive=tmp/"v11.tar.xz"
        archive.write_bytes(payload)
        stage=tmp/"overlay"
        stage.mkdir()
        with tarfile.open(archive, "r:xz") as tar:
            for member in tar.getmembers():
                path=Path(member.name)
                if member.issym() or member.islnk() or not member.isfile() or (
                    path.is_absolute() or ".." in path.parts
                ):
                    raise RuntimeError(f"unsafe archive member: {member.name}")
            tar.extractall(stage,filter="data")
        subprocess.run([sys.executable,str(stage/"scripts/install_v11_overlay.py"),
                        "--repo",str(BASE)],check=True)
        shutil.copy2(stage/"scripts/install_v11_overlay.py",
                     BASE/"scripts/install_v11_overlay.py")
        shutil.copy2(stage/"tests/test_v11.py",BASE/"tests/test_v11.py")
        shutil.copy2(stage/"README.md",BASE/"docs/V11_B6_OVERLAY.md")
    print("B6 source integrated; run full tests before merge.")

if __name__=="__main__":
    main()
