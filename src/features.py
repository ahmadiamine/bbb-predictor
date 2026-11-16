"""
Turn a molecule into eight numbers a model can learn from.

The blood-brain barrier is a sheet of endothelial cells with unusually tight
junctions. Almost nothing crosses between the cells, so a molecule has to pass
straight through them - which means dissolving into a lipid membrane, crossing
it, and coming out the other side. That physical picture predicts which
descriptors should matter:

  tpsa              polar surface area. Every polar group drags a shell of water
                    that has to be stripped off before entry. The single most
                    cited BBB descriptor; above ~90 A^2 penetration is unlikely.
  logp              fat/water partitioning. Too polar and it never enters the
                    membrane; too greasy and it never leaves it.
  h_donors          N-H and O-H groups hydrogen bond strongly to water and are
                    expensive to desolvate. Usually the second strongest signal.
  mol_weight        big molecules diffuse slowly; the practical ceiling is ~450.
  h_acceptors       N and O atoms, polar for the same reason as donors
  rotatable_bonds   flexibility, which costs entropy on entry
  aromatic_rings    flat hydrophobic surface, generally helpful
  fraction_csp3     how three-dimensional the carbon skeleton is

Every function goes through rdMolDescriptors, which is the same C++ code RDKit.js
runs in the browser. That is what lets docs/predict.js reproduce these numbers
exactly; tests/test_browser_parity.py proves it.
"""
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

FEATURE_NAMES = [
    "tpsa", "logp", "h_donors", "mol_weight",
    "h_acceptors", "rotatable_bonds", "aromatic_rings", "fraction_csp3",
]

LABELS = {
    "tpsa": "Polar surface area",
    "logp": "logP (greasiness)",
    "h_donors": "H-bond donors",
    "mol_weight": "Molecular weight",
    "h_acceptors": "H-bond acceptors",
    "rotatable_bonds": "Rotatable bonds",
    "aromatic_rings": "Aromatic rings",
    "fraction_csp3": "Fraction sp3 carbons",
}

UNITS = {"tpsa": "A^2", "mol_weight": "Da"}


def parse(smiles):
    """SMILES string -> RDKit molecule, or None if it is not valid chemistry."""
    return Chem.MolFromSmiles(str(smiles).strip())


def featurize_mol(mol):
    """Molecule -> the eight numbers, as a dict."""
    return {
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
        "h_donors": float(rdMolDescriptors.CalcNumHBD(mol)),
        "mol_weight": rdMolDescriptors.CalcExactMolWt(mol),
        "h_acceptors": float(rdMolDescriptors.CalcNumHBA(mol)),
        "rotatable_bonds": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "aromatic_rings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def featurize(smiles):
    """SMILES -> the eight numbers. Raises ValueError on invalid chemistry."""
    mol = parse(smiles)
    if mol is None:
        raise ValueError(f"not a valid SMILES string: {smiles!r}")
    return featurize_mol(mol)


# ------------------------------------------------------------ the rival rule

CNS_RULES = [
    ("TPSA <= 90",        lambda f: f["tpsa"] <= 90),
    ("MW <= 450",         lambda f: f["mol_weight"] <= 450),
    ("1 <= logP <= 4",    lambda f: 1.0 <= f["logp"] <= 4.0),
    ("H-bond donors <= 3", lambda f: f["h_donors"] <= 3),
]


def cns_rule(features):
    """
    The medicinal chemistry rule of thumb this project is trying to beat.

    Four cutoffs that CNS drug discovery teams have used for decades. A molecule
    "passes" if it breaks at most one. It is crude, it has no training data
    behind it, and it is the benchmark any model here has to clear to be worth
    anything.
    """
    broken = [name for name, test in CNS_RULES if not test(features)]
    return {"violations": len(broken), "broken": broken,
            "passes": len(broken) <= 1}


if __name__ == "__main__":
    examples = [
        ("Caffeine (enters brain)", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("Diazepam (enters brain)", "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1"),
        ("Dopamine (blocked)",      "NCCc1ccc(O)c(O)c1"),
        ("Sucrose (blocked)",
         "OC[C@H]1O[C@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
         "[C@@H](O)[C@@H]1O"),
    ]
    for name, smi in examples:
        f = featurize(smi)
        rule = cns_rule(f)
        print(f"{name:26} TPSA={f['tpsa']:6.1f}  logP={f['logp']:5.2f}  "
              f"HBD={f['h_donors']:.0f}  rule violations={rule['violations']}")
