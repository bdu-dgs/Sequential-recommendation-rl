"""Download KuaiRec 2.0 (CC BY-SA 4.0) from Zenodo and unpack it under data/raw/.

Source: https://github.com/chongminggao/KuaiRec  (Zenodo record 18164998)
"""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

from seqrec_rl import DATA_DIR

URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip?download=1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DATA_DIR / "raw")
    args = ap.parse_args()
    dl_dir = args.out / "kuairec_download"
    dl_dir.mkdir(parents=True, exist_ok=True)
    zpath = dl_dir / "KuaiRec.zip"
    if not zpath.exists():
        print(f"downloading {URL} -> {zpath}")
        urllib.request.urlretrieve(URL, zpath)
    print("sha256", sha256(zpath))
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(args.out)
    print("extracted to", args.out)


if __name__ == "__main__":
    main()
