"""Restore the pinned IBM CSV and verify its bytes before replacing local data."""

import hashlib
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv"
)
EXPECTED = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def main():
    with urllib.request.urlopen(URL, timeout=30) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != EXPECTED:
        raise ValueError("Dataset checksum mismatch; existing file was not changed.")
    destination = Path(__file__).resolve().parents[1] / "data" / "telco.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    print(f"Verified dataset saved to {destination}")


if __name__ == "__main__":
    main()
