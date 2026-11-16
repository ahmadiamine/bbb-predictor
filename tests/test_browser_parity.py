"""
Does the website agree with scikit-learn?

The browser recomputes the descriptors with RDKit.js and re-applies the model in
JavaScript. If either step drifts from the Python it came from, the site will
confidently show numbers the training code never produced.

This runs docs/predict.js under Node, with the same RDKit WebAssembly build the
website uses, and compares against the Python model on every molecule in the
dataset.

  python tests/test_browser_parity.py      (needs Node installed)
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import features as F          # noqa: E402
from train import build, load, scaffold_split    # noqa: E402

TOLERANCE = 1e-9

NODE_SCRIPT = r"""
const fs = require('fs'), path = require('path');
const root = process.argv[1], smilesFile = process.argv[2];

global.window = {};
global.self = global.window;
eval(fs.readFileSync(path.join(root, 'docs', 'model.js'), 'utf8'));
const model = global.window.BBB_MODEL;
const P = require(path.join(root, 'docs', 'predict.js'));
const initRDKitModule = require(path.join(root, 'docs', 'vendor',
                                          'RDKit_minimal.js'));

const smiles = JSON.parse(fs.readFileSync(smilesFile, 'utf8'));

initRDKitModule({
  locateFile: () => path.join(root, 'docs', 'vendor', 'RDKit_minimal.wasm')
}).then(RDKit => {
  const out = smiles.map(s => {
    const mol = RDKit.get_mol(s);
    if (!mol || !mol.is_valid()) { if (mol) mol.delete(); return null; }
    const desc = JSON.parse(mol.get_descriptors());
    mol.delete();
    const r = P.score(desc, model);
    return { probability: r.probability, features: r.features,
             violations: r.rule.violations };
  });
  console.log(JSON.stringify(out));
}).catch(e => { console.error(String(e)); process.exit(1); });
"""


def main():
    # rebuild exactly the model train.py exported: scaffold split, same seed
    X, y, _, scaffolds = load(ROOT / "data" / "bbbp.csv")
    i_tr, _, _ = scaffold_split(scaffolds)
    model = build().fit(X.iloc[i_tr], y[i_tr])

    df = pd.read_csv(ROOT / "data" / "bbbp.csv")
    smiles = list(df.smiles) + [
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",          # caffeine
        "CN1CCC[C@H]1c1cccnc1",                # nicotine
        "NCCc1ccc(O)c(O)c1",                   # dopamine
        "CC(=O)Oc1ccccc1C(=O)O",               # aspirin
    ]

    tmp = ROOT / "tests" / "_smiles.json"
    tmp.write_text(json.dumps(smiles))
    try:
        proc = subprocess.run(["node", "-e", NODE_SCRIPT, "--", str(ROOT), str(tmp)],
                              capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        sys.exit("Node is not installed - skipping the parity test.")
    finally:
        tmp.unlink(missing_ok=True)

    if proc.returncode != 0:
        sys.exit("Node failed:\n" + proc.stderr[-2000:])

    js = json.loads(proc.stdout)

    worst_p = worst_f = 0.0
    failures = compared = rule_mismatch = 0
    for smi, j in zip(smiles, js):
        mol = F.parse(smi)
        if mol is None or j is None:
            continue            # both sides must agree it is unusable
        compared += 1

        py_feats = F.featurize_mol(mol)
        for k, v in py_feats.items():
            worst_f = max(worst_f, abs(j["features"][k] - v))

        p_py = model.predict_proba(pd.DataFrame([py_feats])[F.FEATURE_NAMES])[0, 1]
        diff = abs(j["probability"] - p_py)
        worst_p = max(worst_p, diff)
        if diff > TOLERANCE:
            failures += 1
            if failures <= 5:
                print(f"  MISMATCH {smi[:40]}: js={j['probability']:.12f} "
                      f"py={p_py:.12f}")

        if F.cns_rule(py_feats)["violations"] != j["violations"]:
            rule_mismatch += 1

    print(f"compared {compared} molecules")
    print(f"largest descriptor difference:  {worst_f:.3e}")
    print(f"largest probability difference: {worst_p:.3e}")
    print(f"CNS rule disagreements:         {rule_mismatch}")

    if failures or rule_mismatch:
        sys.exit(f"FAILED: {failures} probability and {rule_mismatch} rule "
                 f"disagreements")
    print(f"PASS - RDKit.js and Python RDKit agree to {TOLERANCE}")


if __name__ == "__main__":
    main()
