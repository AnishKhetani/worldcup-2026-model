"""Mirror the two free World Cup data sources into data/raw/.

Source 1: FIFA-World-Cup-2026-Dataset (CC0-1.0) relational CSVs -> data/raw/wc2026/
Source 2: martj42/international_results (CC0) results.csv        -> data/raw/intl/

Bytes are written verbatim so the rest of the pipeline is offline and reproducible.
Both upstreams update daily, so re-running with --force refreshes the mirror. Run
wc_validate.py afterwards to check what was pulled.
"""
from __future__ import annotations

import sys

import requests

from wc_config import (HTTP_HEADERS, HTTP_TIMEOUT, WC_BASE_URL, WC_CSV_FILES,
                       WC_RAW_DIR)

_S = requests.Session()
_S.headers.update(HTTP_HEADERS)


def _get(url: str) -> bytes | None:
    try:
        r = _S.get(url, timeout=HTTP_TIMEOUT)
    except Exception as e:  # network error -- report, don't crash the whole run
        print(f"    error ({type(e).__name__})")
        return None
    if r.status_code != 200:
        print(f"    http {r.status_code}")
        return None
    return r.content


def fetch_wc_csvs(force: bool = False) -> int:
    print("--- Source 1: FIFA-World-Cup-2026-Dataset (CC0-1.0) ---")
    got = 0
    for fname in WC_CSV_FILES:
        dest = WC_RAW_DIR / fname
        if dest.exists() and not force:
            print(f"  {fname:<26} cached ({dest.stat().st_size:,} bytes)")
            got += 1
            continue
        body = _get(f"{WC_BASE_URL}/{fname}")
        if body is None or len(body) < 10 or b"," not in body[:2000]:
            print(f"  {fname:<26} MISSING/empty")
            continue
        dest.write_bytes(body)
        print(f"  {fname:<26} ok ({len(body):,} bytes)")
        got += 1
    return got


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    force = "--force" in argv
    print("=" * 70)
    print("FETCH World Cup 2026 data (CC0 relational CSVs + international results)")
    print("=" * 70)
    n_csv = fetch_wc_csvs(force=force)

    print("\n--- Source 2: all-international results history (martj42, CC0) ---")
    try:
        from wc_train import RESULTS_CSV, download
        download(force=force)
        ok_intl = RESULTS_CSV.exists()
        print(f"  results.csv                {'present' if ok_intl else 'MISSING'} "
              f"({RESULTS_CSV.stat().st_size:,} bytes)" if ok_intl else "")
    except Exception as e:
        ok_intl = False
        print(f"  results.csv                error ({type(e).__name__})")

    print("\n--- Source 3: Fjelstul World Cup Database (CC-BY-SA 4.0) ---")
    try:
        from wc_knockout import FJELSTUL_CSV, download_fjelstul
        download_fjelstul(force=force)
        ok_fj = FJELSTUL_CSV.exists()
        print(f"  matches.csv                {'present' if ok_fj else 'MISSING'} "
              f"({FJELSTUL_CSV.stat().st_size:,} bytes)" if ok_fj else "")
    except Exception as e:
        ok_fj = False
        print(f"  matches.csv                error ({type(e).__name__})")

    print(f"\nDone. {n_csv}/{len(WC_CSV_FILES)} CC0 CSVs present; "
          f"intl history {'present' if ok_intl else 'MISSING'}; "
          f"Fjelstul WC DB {'present' if ok_fj else 'MISSING'}.")
    return 0 if (n_csv and ok_intl and ok_fj) else 1


if __name__ == "__main__":
    sys.exit(main())
