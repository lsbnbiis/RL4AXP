"""
Three-tier negative-sample dataset cleaning (SQA_AMP_RL.md §四).

Downloads MIC-labeled peptides for a target species from the DBAASP REST API
(https://dbaasp.org, OpenAPI spec at /v3/api-docs) and builds a 1:1 balanced
positive/negative dataset:

  positive          : MIC <= 8 µg/mL
  negative, tier 1   : MIC > 64 µg/mL          (naturally inactive)
  negative, tier 2   : positive sequences with shuffled residues, sampled to
                       fill the gap so tier1 + tier2 == len(positive)

Usage
-----
    python build_amp_dataset.py --target-species "Staphylococcus aureus" \
        --out balanced_amp_dataset.csv

    # quick smoke test against a handful of peptides
    python build_amp_dataset.py --limit 20 --out /tmp/test_balanced.csv
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import List, Optional

import pandas as pd
import requests

DBAASP_BASE = "https://dbaasp.org"


def fetch_dbaasp_ids(
    session: requests.Session,
    target_species: str,
    complexity: str = "monomer",
    page_size: int = 200,
    limit: Optional[int] = None,
) -> List[int]:
    """Page through GET /peptides filtered by target species, return peptide ids."""
    ids: List[int] = []
    offset = 0
    while True:
        resp = session.get(
            f"{DBAASP_BASE}/peptides",
            params={
                "targetSpecies.value": target_species,
                "complexity.value": complexity,
                "limit": page_size,
                "offset": offset,
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        batch = payload.get("data", [])
        if not batch:
            break

        ids.extend(item["id"] for item in batch)
        if limit is not None and len(ids) >= limit:
            return ids[:limit]

        offset += page_size
        if offset >= payload.get("totalCount", 0):
            break

    return ids


def fetch_peptide_detail(session: requests.Session, peptide_id: int) -> dict:
    resp = session.get(f"{DBAASP_BASE}/peptides/{peptide_id}", timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_target_mic(detail: dict, target_species: str) -> Optional[float]:
    """Most conservative (lowest) MIC value recorded against `target_species`."""
    mics = [
        act["activity"]
        for act in detail.get("targetActivities", [])
        if act.get("targetSpecies", {}).get("name") == target_species
        and act.get("activityMeasureGroup", {}).get("name") == "MIC"
        and act.get("activity") is not None
    ]
    return min(mics) if mics else None


def fetch_dbaasp_records(
    target_species: str,
    complexity: str = "monomer",
    limit: Optional[int] = None,
    request_delay: float = 0.05,
) -> pd.DataFrame:
    """Fetch id -> sequence -> MIC for every matching DBAASP peptide."""
    session = requests.Session()
    ids = fetch_dbaasp_ids(session, target_species, complexity=complexity, limit=limit)

    records = []
    for peptide_id in ids:
        detail = fetch_peptide_detail(session, peptide_id)
        sequence = detail.get("sequence", "")
        mic = extract_target_mic(detail, target_species)
        if sequence and mic is not None:
            records.append({"id": peptide_id, "sequence": sequence, "MIC": mic})
        time.sleep(request_delay)

    return pd.DataFrame.from_records(records)


def build_balanced_dataset(df: pd.DataFrame, seed: int = 3407) -> pd.DataFrame:
    """1:1 balanced positive/negative dataset (SQA_AMP_RL.md §四.2)."""
    rng = random.Random(seed)

    pos_dataset = df[df["MIC"] <= 8]
    neg_from_db = df[df["MIC"] > 64]

    needed_shuffled = max(0, len(pos_dataset) - len(neg_from_db))
    neg_shuffled = []

    sample_n = min(needed_shuffled, len(pos_dataset))
    for _, row in pos_dataset.sample(n=sample_n, random_state=seed).iterrows():
        seq_list = list(row["sequence"])
        rng.shuffle(seq_list)
        neg_shuffled.append("".join(seq_list))

    neg_all = list(neg_from_db["sequence"]) + neg_shuffled
    balanced_df = pd.DataFrame({
        "sequence": list(pos_dataset["sequence"]) + neg_all,
        "label": [1] * len(pos_dataset) + [0] * len(neg_all),
    })
    return balanced_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-species", default="Staphylococcus aureus")
    parser.add_argument("--complexity", default="monomer")
    parser.add_argument("--limit", type=int, default=None, help="cap peptides fetched (testing)")
    parser.add_argument("--cache", default=None, help="path to cache raw fetched records as JSON")
    parser.add_argument("--out", default="balanced_amp_dataset.csv")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if args.cache and __import__("os").path.exists(args.cache):
        print(f"Loading cached records from {args.cache}")
        df = pd.read_json(args.cache)
    else:
        print(f"Fetching DBAASP records for target species '{args.target_species}'...")
        df = fetch_dbaasp_records(args.target_species, complexity=args.complexity, limit=args.limit)
        print(f"Fetched {len(df)} peptides with usable sequence + MIC.")
        if args.cache:
            df.to_json(args.cache, orient="records")
            print(f"Cached raw records to {args.cache}")

    balanced_df = build_balanced_dataset(df, seed=args.seed)
    balanced_df.to_csv(args.out, index=False)
    print(f"Wrote {len(balanced_df)} rows ({(balanced_df['label'] == 1).sum()} positive / "
          f"{(balanced_df['label'] == 0).sum()} negative) to {args.out}")


if __name__ == "__main__":
    main()
