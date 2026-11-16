/*
 * The model, ported to JavaScript.
 *
 * RDKit.js computes the descriptors in the browser using the same C++ code as
 * Python RDKit, so the only thing that has to be ported by hand is the
 * arithmetic: map RDKit's descriptor names onto our eight features, apply the
 * StandardScaler, then the logistic regression.
 *
 * Kept in its own file so tests/test_browser_parity.py can run it under Node
 * and check every number against scikit-learn.
 *
 * Works in the browser (window.bbbPredict) and in Node (module.exports).
 */
(function (root) {
  "use strict";

  /* our feature name -> the key RDKit.js uses in get_descriptors() */
  var RDKIT_KEYS = {
    tpsa: "tpsa",
    logp: "CrippenClogP",
    h_donors: "NumHBD",
    mol_weight: "exactmw",
    h_acceptors: "NumHBA",
    rotatable_bonds: "NumRotatableBonds",
    aromatic_rings: "NumAromaticRings",
    fraction_csp3: "FractionCSP3"
  };

  /* the CNS rule of thumb, mirroring CNS_RULES in src/features.py */
  var CNS_RULES = [
    { label: "TPSA ≤ 90", test: function (f) { return f.tpsa <= 90; } },
    { label: "MW ≤ 450", test: function (f) { return f.mol_weight <= 450; } },
    { label: "logP 1–4", test: function (f) { return f.logp >= 1 && f.logp <= 4; } },
    { label: "donors ≤ 3", test: function (f) { return f.h_donors <= 3; } }
  ];

  /* RDKit descriptor JSON -> the eight numbers src/features.py produces */
  function featurize(descriptors, model) {
    var f = {};
    for (var i = 0; i < model.feature_names.length; i++) {
      var name = model.feature_names[i];
      f[name] = descriptors[RDKIT_KEYS[name]];
    }
    return f;
  }

  /* features -> probability, plus each descriptor's pull on the log-odds */
  function predict(f, model) {
    var logit = model.intercept;
    var parts = [];
    for (var i = 0; i < model.feature_names.length; i++) {
      var name = model.feature_names[i];
      var z = (f[name] - model.mean[i]) / model.scale[i];   // StandardScaler
      var contribution = model.coef[i] * z;
      logit += contribution;
      parts.push({ name: name, value: f[name], contribution: contribution });
    }
    return { probability: 1 / (1 + Math.exp(-logit)), logit: logit, parts: parts };
  }

  function cnsRule(f) {
    var broken = [];
    var checks = CNS_RULES.map(function (r) {
      var ok = r.test(f);
      if (!ok) broken.push(r.label);
      return { label: r.label, ok: ok };
    });
    return { checks: checks, violations: broken.length, passes: broken.length <= 1 };
  }

  /* convenience: descriptors in, everything out */
  function score(descriptors, model) {
    var f = featurize(descriptors, model);
    var out = predict(f, model);
    out.features = f;
    out.rule = cnsRule(f);
    return out;
  }

  var api = { RDKIT_KEYS: RDKIT_KEYS, CNS_RULES: CNS_RULES,
              featurize: featurize, predict: predict, cnsRule: cnsRule,
              score: score };

  if (typeof module === "object" && module.exports) module.exports = api;
  else root.bbbPredict = api;
})(typeof self !== "undefined" ? self : this);
