"""`slic data` — download the released task sets from the Hugging Face Hub (dataset `<org>/slic-data`: one folder per task
family plus `ptm-cards/`). Only the requested folders are fetched.

  slic data --family standard            -> data/standard/{train,validation,test_iid,test_ood}.parquet
  slic data --family hard                -> data/hard/{train,validation,test_iid,test_ood}.parquet
  slic data --family nodes               -> data/nodes/{validation,test_ood}_{22nm,32nm,45nm,90nm,130nm}.parquet
                                            (the standard validation / test_ood tasks rebuilt at the other nodes; needs --cards)
  slic data --cards                      -> circuit/dataset/{22nm_bulk_v2,32nm_HP,45nm_HP,65nm_bulk,90nm_bulk,130nm_bulk}.scs
                                            (predictive technology model cards for the other nodes; 65 nm ships with the repo)
  slic data --topologies                 -> data/topologies/<topology>/{schematic.svg,schematic.png,netlist.scs} + topologies.csv
                                            (the browser UI shows the schematic of the selected circuit)
  --revision <commit>                    pins the dataset version (default: the latest)

The hub organisation is read from --org or SLIC_HF_ORG.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from ._common import REPO_ROOT, ConfigError

DATA_REPO = "slic-data"       # dataset repository: <family>/ folders and ptm-cards/
DATASETS = {"standard": "standard", "hard": "hard", "nodes": "nodes"}
CARDS = "ptm-cards"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slic data", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(DATASETS), action="append", default=[])
    ap.add_argument("--cards", action="store_true", help="technology model cards for the 22-130 nm node sweep -> circuit/dataset/")
    ap.add_argument("--topologies", action="store_true", help="schematics + netlists of every topology -> data/topologies/")
    ap.add_argument("--revision", default=None, help="dataset revision (commit) to download; default: the latest")
    ap.add_argument("--org", default=os.environ.get("SLIC_HF_ORG", "slic-agent"), help="Hugging Face user/org that hosts the release")
    ap.add_argument("--out", default=str(REPO_ROOT), help="repo root under which data/ is created")
    a = ap.parse_args(argv)
    if not a.family and not a.cards and not a.topologies:
        ap.error("nothing to fetch: pass --family, --cards and / or --topologies")
    if not a.org:
        raise ConfigError("set --org or SLIC_HF_ORG to the Hugging Face organisation hosting the release")
    from huggingface_hub import snapshot_download

    out = Path(a.out)
    for fam in a.family:
        folder = DATASETS[fam]
        snapshot_download(repo_id=f"{a.org}/{DATA_REPO}", repo_type="dataset", local_dir=str(out / "data"),
                          allow_patterns=[f"{folder}/*"], revision=a.revision)
        print(f"[slic] {fam} task sets -> {out / 'data' / folder}")
    if a.cards:
        dest = out / "circuit" / "dataset"
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(repo_id=f"{a.org}/{DATA_REPO}", repo_type="dataset", local_dir=tmp,
                              allow_patterns=[f"{CARDS}/*.scs"], revision=a.revision)
            cards = sorted((Path(tmp) / CARDS).glob("*.scs"))
            if not cards:
                raise ConfigError(f"no technology cards found under {CARDS}/ in {a.org}/{DATA_REPO}")
            for c in cards:
                shutil.copy2(c, dest / c.name)
        print(f"[slic] technology model cards ({len(cards)}) -> {dest}")
    if a.topologies:
        snapshot_download(repo_id=f"{a.org}/{DATA_REPO}", repo_type="dataset", local_dir=str(out / "data"),
                          allow_patterns=["topologies/*/*", "topologies.csv"], revision=a.revision)
        n = len(list((out / "data" / "topologies").glob("*/schematic.svg")))
        if n == 0:
            raise ConfigError(f"no schematics found under topologies/ in {a.org}/{DATA_REPO}")
        print(f"[slic] schematics + netlists of {n} topologies -> {out / 'data' / 'topologies'}")
    return 0
