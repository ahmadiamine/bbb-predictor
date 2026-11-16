"""
Build the dataset from the public BBBP benchmark -> data/bbbp.csv

  python src/fetch_data.py

SOURCE: the Blood-Brain Barrier Penetration (BBBP) dataset, part of MoleculeNet.
Around 2,050 compounds, each labelled by whether it was measured to cross the
blood-brain barrier. The labels come from the published pharmacology literature,
not from a rule or a simulation.

  label 1  penetrates the barrier  (reaches the brain)
  label 0  does not

Cleaning, applied identically to every molecule:
  1. keep only the largest fragment   - the raw file stores drugs as formulated,
                                        so propranolol arrives as its
                                        hydrochloride salt, "...[Cl]"
  2. neutralise charges               - same molecule, one representation
  3. re-canonicalise the SMILES       - so duplicates become detectable
  4. drop unparseable records         - a handful of entries are malformed
  5. drop duplicates and conflicts    - a molecule appearing twice with opposite
                                        labels is unusable, so both copies go

Step 5 matters more than it looks: a duplicate that survives into both the
training and test sets hands the model a free correct answer.
"""
import csv
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
URL = ("https://raw.githubusercontent.com/GLambard/Molecules_Dataset_Collection/"
       "master/latest/BBBP.csv")

ORGANIC = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
MIN_MW, MAX_MW = 50.0, 900.0
MIN_HEAVY = 4

_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def clean(smiles):
    """One raw SMILES in, one canonical SMILES out (or None if unusable)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = _chooser.choose(mol)
    mol = _uncharger.uncharge(mol)
    if mol is None or mol.GetNumHeavyAtoms() < MIN_HEAVY:
        return None
    if any(a.GetSymbol() not in ORGANIC for a in mol.GetAtoms()):
        return None
    if not (MIN_MW <= Descriptors.MolWt(mol) <= MAX_MW):
        return None
    return Chem.MolToSmiles(mol)


def download(dest):
    if dest.exists():
        print(f"  using cached {dest.name}")
        return dest
    print(f"  downloading {dest.name} ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "bbb-predictor/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, dest.open("wb") as fh:
        fh.write(r.read())
    return dest


def main():
    cache = ROOT / "data" / "raw"
    cache.mkdir(parents=True, exist_ok=True)

    print("Source")
    try:
        path = download(cache / "BBBP.csv")
    except Exception as exc:                                   # noqa: BLE001
        sys.exit(f"download failed ({exc}).\nThe file is on GitHub; if you are "
                 "offline, copy BBBP.csv into data/raw/ by hand.")

    raw = list(csv.DictReader(path.open()))
    print(f"\nread {len(raw)} records")

    unparseable = 0
    by_smiles = defaultdict(list)          # canonical smiles -> [(name, label)]
    for r in raw:
        c = clean(r["smiles"])
        if c is None:
            unparseable += 1
            continue
        by_smiles[c].append((r.get("name", "").strip(), int(r["p_np"])))

    print(f"  {unparseable} records dropped as unparseable or out of scope")

    rows, conflicts, dupes = [], 0, 0
    for smi, entries in by_smiles.items():
        labels = {lab for _, lab in entries}
        if len(labels) > 1:                # same molecule, contradictory labels
            conflicts += 1
            continue
        if len(entries) > 1:
            dupes += len(entries) - 1
        name = entries[0][0] or "unnamed"
        rows.append({"name": name, "smiles": smi, "label": entries[0][1]})

    print(f"  {dupes} duplicate entries collapsed")
    print(f"  {conflicts} molecules dropped for contradictory labels")

    out = ROOT / "data" / "bbbp.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "smiles", "label"])
        w.writeheader()
        w.writerows(rows)

    n_pos = sum(r["label"] for r in rows)
    print(f"\nwrote {out}")
    print(f"  {len(rows)} molecules: {n_pos} penetrate / {len(rows) - n_pos} do not "
          f"({100 * n_pos / len(rows):.0f}% positive - the classes are imbalanced)")
    print("\nnow run:  python src/train.py")


if __name__ == "__main__":
    main()
