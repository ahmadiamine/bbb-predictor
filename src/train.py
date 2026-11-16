"""
Train the classifier, split it two different ways, and export it for the website.

  python src/train.py
  python src/train.py --split random     (only the optimistic split)

The headline of this project is not the score. It is the GAP between two scores:

  random split    molecules shuffled at random into train and test. Analogues of
                  the same drug land on both sides, so the model is often asked
                  about a molecule whose near-twin it has already seen.
  scaffold split  molecules grouped by their Murcko scaffold - their core ring
                  system - with whole groups assigned to one side. The test set
                  is then chemistry the model has never encountered.

Real drug discovery is always the second case: you are asking about a molecule
nobody has made yet. A project that reports only the random-split number is
reporting a number that does not transfer.

Two rivals, both evaluated on the same scaffold split:
  CNS rule of thumb   four hand-written cutoffs, no training
  TPSA alone          one descriptor, the most cited BBB predictor

Produces results/metrics.txt, results/model.json and docs/model.js.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import features as F

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TEST_FRACTION = 0.25


def load(path):
    """Read the CSV, compute descriptors and the Murcko scaffold of each molecule."""
    df = pd.read_csv(path)
    rows, labels, names, scaffolds = [], [], [], []
    for _, r in df.iterrows():
        mol = F.parse(r["smiles"])
        if mol is None:
            continue
        rows.append(F.featurize_mol(mol))
        labels.append(int(r["label"]))
        names.append(str(r["name"]))
        try:
            scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:                                      # noqa: BLE001
            scaffolds.append("")                               # acyclic molecule
    X = pd.DataFrame(rows, columns=F.FEATURE_NAMES)
    return X, np.array(labels), names, scaffolds


def scaffold_split(scaffolds, test_fraction=TEST_FRACTION):
    """
    Group molecules by core ring system, then assign whole groups to train or
    test. Largest groups go to train first, which is the standard convention -
    it leaves the test set made of rarer, more varied chemistry.
    """
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)

    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n_test_target = int(len(scaffolds) * test_fraction)

    train_idx, test_idx = [], []
    for group in ordered:
        if len(test_idx) + len(group) <= n_test_target:
            test_idx.extend(group)
        else:
            train_idx.extend(group)
    return np.array(sorted(train_idx)), np.array(sorted(test_idx)), len(groups)


def build():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED,
                                   class_weight="balanced")),
    ])


def evaluate(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "auc": roc_auc_score(y_te, prob),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "prob": prob, "pred": pred, "model": model,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "bbbp.csv"))
    ap.add_argument("--split", choices=["both", "random", "scaffold"],
                    default="both")
    args = ap.parse_args()

    # -------------------------------------------------------------- 1. load
    X, y, names, scaffolds = load(args.data)
    print(f"\ndata: {len(X)} molecules, {y.sum()} penetrate / "
          f"{len(y) - y.sum()} do not ({100 * y.mean():.0f}% positive)")
    print(f"descriptors: {', '.join(F.FEATURE_NAMES)}")

    # ----------------------------------------------- 2. the two ways to split
    i_tr_r, i_te_r = train_test_split(np.arange(len(X)), test_size=TEST_FRACTION,
                                      random_state=SEED, stratify=y)
    i_tr_s, i_te_s, n_scaffolds = scaffold_split(scaffolds)
    print(f"\n{n_scaffolds} distinct Murcko scaffolds among {len(X)} molecules")
    print(f"  random split:   {len(i_tr_r)} train / {len(i_te_r)} test")
    print(f"  scaffold split: {len(i_tr_s)} train / {len(i_te_s)} test "
          f"(no scaffold appears on both sides)")

    results = {}
    for tag, (i_tr, i_te) in {"random": (i_tr_r, i_te_r),
                              "scaffold": (i_tr_s, i_te_s)}.items():
        results[tag] = evaluate(build(), X.iloc[i_tr], y[i_tr],
                                X.iloc[i_te], y[i_te])

    print("\nsame model, same descriptors, two different splits")
    print(f"  {'random split':18} ROC-AUC {results['random']['auc']:.3f}   "
          f"balanced accuracy {results['random']['balanced_accuracy']:.3f}")
    print(f"  {'scaffold split':18} ROC-AUC {results['scaffold']['auc']:.3f}   "
          f"balanced accuracy {results['scaffold']['balanced_accuracy']:.3f}")
    gap = results["random"]["auc"] - results["scaffold"]["auc"]
    print(f"  the gap: {gap:+.3f} ROC-AUC of pure optimism")

    # ------------------------------------- 3. honest evaluation, from here on
    i_tr, i_te = i_tr_s, i_te_s
    main_res = results["scaffold"]
    model = main_res["model"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc = cross_val_score(build(), X.iloc[i_tr], y[i_tr], cv=cv,
                             scoring="roc_auc")
    print(f"\n5-fold CV within the training set: "
          f"{cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

    # --------------------------------------------------------- 4. the rivals
    feats_te = [X.iloc[i].to_dict() for i in i_te]
    rule_violations = np.array([F.cns_rule(f)["violations"] for f in feats_te])
    rule_auc = roc_auc_score(y[i_te], -rule_violations)
    rule_bacc = balanced_accuracy_score(y[i_te], (rule_violations <= 1).astype(int))
    tpsa_auc = roc_auc_score(y[i_te], -X.iloc[i_te]["tpsa"])

    print("\nhead to head, all on the scaffold split")
    print(f"  {'this model (8 descriptors)':30} ROC-AUC {main_res['auc']:.3f}   "
          f"balanced accuracy {main_res['balanced_accuracy']:.3f}")
    print(f"  {'TPSA alone (1 descriptor)':30} ROC-AUC {tpsa_auc:.3f}")
    print(f"  {'CNS rule of thumb':30} ROC-AUC {rule_auc:.3f}   "
          f"balanced accuracy {rule_bacc:.3f}")

    print("\n" + classification_report(y[i_te], main_res["pred"],
                                       target_names=["blocked", "penetrates"]))
    tn, fp, fn, tp = confusion_matrix(y[i_te], main_res["pred"]).ravel()
    print(f"confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")

    # ------------------------------------------ 5. what did it actually learn
    coefs = model.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    print("\ndescriptor weights (standardised; + pushes towards penetrates)")
    for i in order:
        print(f"  {F.LABELS[F.FEATURE_NAMES[i]]:22} {coefs[i]:+.3f}")

    # -------------------------------------------- 6. sanity check on knowns
    print("\nsanity check on molecules held out by the scaffold split or absent")
    checks = [
        ("Caffeine (enters brain)",  "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("Diazepam (enters brain)",  "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1"),
        ("Nicotine (enters brain)",  "CN1CCC[C@H]1c1cccnc1"),
        ("Dopamine (blocked)",       "NCCc1ccc(O)c(O)c1"),
        ("Sucrose (blocked)",
         "OC[C@H]1O[C@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
         "[C@@H](O)[C@@H]1O"),
    ]
    for label, smi in checks:
        f = F.featurize(smi)
        p = model.predict_proba(pd.DataFrame([f])[F.FEATURE_NAMES])[0, 1]
        print(f"  {label:26} P(penetrates) = {p:.2f}")

    # ------------------------------------------------------------- 7. export
    scaler = model.named_steps["scale"]
    payload = {
        "feature_names": F.FEATURE_NAMES,
        "labels": F.LABELS,
        "units": F.UNITS,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": coefs.tolist(),
        "intercept": float(model.named_steps["clf"].intercept_[0]),
        "metrics": {
            "auc": float(main_res["auc"]),
            "balanced_accuracy": float(main_res["balanced_accuracy"]),
            "random_auc": float(results["random"]["auc"]),
            "gap": float(gap),
            "cv_auc": float(cv_auc.mean()), "cv_auc_std": float(cv_auc.std()),
            "tpsa_auc": float(tpsa_auc), "rule_auc": float(rule_auc),
            "n_train": int(len(i_tr)), "n_test": int(len(i_te)),
            "n_total": int(len(X)), "n_scaffolds": int(n_scaffolds),
        },
        "scatter": [{"x": float(a), "y": float(b), "l": int(c)} for a, b, c in
                    zip(X["tpsa"], X["logp"], y)],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "model.json").write_text(json.dumps(payload, indent=2))
    (ROOT / "docs" / "model.js").write_text(
        "// Generated by src/train.py - do not edit by hand.\n"
        "window.BBB_MODEL = " + json.dumps(payload) + ";\n")
    (ROOT / "results" / "metrics.txt").write_text(
        f"data file: {args.data}\n"
        f"n = {len(X)} ({y.sum()} penetrate / {len(y) - y.sum()} blocked)\n"
        f"{n_scaffolds} distinct Murcko scaffolds\n\n"
        f"random split    ROC-AUC {results['random']['auc']:.3f}\n"
        f"scaffold split  ROC-AUC {main_res['auc']:.3f}  "
        f"balanced accuracy {main_res['balanced_accuracy']:.3f}\n"
        f"gap             {gap:+.3f}\n\n"
        f"TPSA alone      ROC-AUC {tpsa_auc:.3f}\n"
        f"CNS rule        ROC-AUC {rule_auc:.3f}\n"
        f"5-fold CV       ROC-AUC {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}\n\n"
        + "\n".join(f"{F.FEATURE_NAMES[i]:18} {coefs[i]:+.4f}" for i in order)
        + "\n")
    print("\nwrote results/model.json, results/metrics.txt and docs/model.js")


if __name__ == "__main__":
    main()
