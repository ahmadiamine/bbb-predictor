/*
 * The page.
 *
 * model.js    the trained coefficients, written by src/train.py
 * predict.js  the arithmetic, shared with the Node parity test
 * RDKit.js    chemistry compiled to WebAssembly - parses SMILES, computes the
 *             descriptors, and draws the structure
 *
 * Everything runs locally in the visitor's browser. There is no server.
 */
(function () {
  "use strict";

  var M = window.BBB_MODEL;
  var P = window.bbbPredict;
  var RDKit = null;

  var $ = function (id) { return document.getElementById(id); };
  var box = $("smiles"), alertBox = $("alert"), drawing = $("drawing");
  var formula = $("formula"), word = $("verdict-word"), num = $("verdict-num");
  var fill = $("meter-fill"), whyBody = $("why-body"), plot = $("plot");

  /* ----------------------------------------------------------- formatting */

  function format(name, v) {
    if (name === "tpsa") return v.toFixed(1) + " Å²";
    if (name === "mol_weight") return v.toFixed(1) + " Da";
    if (name === "fraction_csp3") return (v * 100).toFixed(0) + "%";
    if (name === "logp") return (v > 0 ? "+" : "") + v.toFixed(2);
    return String(v);
  }

  function verdictWord(p) {
    if (p >= 0.8) return "Likely reaches the brain";
    if (p >= 0.5) return "Probably reaches the brain";
    if (p >= 0.2) return "Probably blocked";
    return "Likely blocked";
  }

  function renderWhy(parts) {
    var biggest = parts.reduce(function (m, p) {
      return Math.max(m, Math.abs(p.contribution));
    }, 0.001);

    whyBody.innerHTML = "";
    parts.slice().sort(function (a, b) {
      return Math.abs(b.contribution) - Math.abs(a.contribution);
    }).forEach(function (p) {
      var width = (Math.abs(p.contribution) / biggest) * 48;
      var positive = p.contribution >= 0;
      var bar = positive
        ? '<span class="pos" style="width:' + width + '%"></span>'
        : '<span class="neg" style="left:' + (50 - width) + '%;width:' + width + '%"></span>';

      var tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" + M.labels[p.name] + "</td>" +
        "<td>" + format(p.name, p.value) + "</td>" +
        '<td><div class="pull" title="' +
          (positive ? "pushes towards reaching the brain" : "pushes towards blocked") +
        '">' + bar + "</div></td>";
      whyBody.appendChild(tr);
    });
  }

  function renderRule(rule) {
    $("rule-checks").innerHTML = rule.checks.map(function (c) {
      return '<span class="pill ' + (c.ok ? "pill--ok" : "pill--no") + '">' +
             (c.ok ? "✓ " : "✕ ") + c.label + "</span>";
    }).join("");
    $("rule-note").textContent = rule.passes
      ? "Passes the rule of thumb (" + rule.violations + " of 4 broken). The rule "
        + "can only answer yes or no; the model weighs the same properties on a "
        + "sliding scale, so it can say \u201cborderline\u201d."
      : "Fails the rule of thumb (" + rule.violations + " of 4 broken). The rule "
        + "can only answer yes or no; the model weighs the same properties on a "
        + "sliding scale, so it can say \u201cborderline\u201d.";
  }

  /* ---------------------------------------------------------------- plot */

  var W = 440, H = 320, PAD = { t: 16, r: 14, b: 44, l: 48 };
  var xMin = 0, xMax = 240, yMin = -6, yMax = 9;

  function sx(v) {
    v = Math.max(xMin, Math.min(xMax, v));
    return PAD.l + ((v - xMin) / (xMax - xMin)) * (W - PAD.l - PAD.r);
  }
  function sy(v) {
    v = Math.max(yMin, Math.min(yMax, v));
    return H - PAD.b - ((v - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b);
  }
  function el(tag, attrs) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    return e;
  }

  function drawPlotBase() {
    plot.innerHTML = "";
    [0, 60, 120, 180, 240].forEach(function (t) {
      plot.appendChild(el("line", { x1: sx(t), x2: sx(t), y1: PAD.t, y2: H - PAD.b,
                                    stroke: "#eceef2", "stroke-width": 1 }));
      var l = el("text", { x: sx(t), y: H - PAD.b + 15, "text-anchor": "middle",
                           "font-size": 10, fill: "#5b6472",
                           "font-family": "JetBrains Mono, monospace" });
      l.textContent = t;
      plot.appendChild(l);
    });
    [-5, 0, 5].forEach(function (t) {
      plot.appendChild(el("line", { x1: PAD.l, x2: W - PAD.r, y1: sy(t), y2: sy(t),
                                    stroke: "#eceef2", "stroke-width": 1 }));
      var l = el("text", { x: PAD.l - 7, y: sy(t) + 3.5, "text-anchor": "end",
                           "font-size": 10, fill: "#5b6472",
                           "font-family": "JetBrains Mono, monospace" });
      l.textContent = t;
      plot.appendChild(l);
    });

    M.scatter.forEach(function (d) {
      plot.appendChild(el("circle", { cx: sx(d.x), cy: sy(d.y), r: 2.6,
                                      fill: d.l ? "#1d4ed8" : "#b45309",
                                      "fill-opacity": 0.38 }));
    });

    // the textbook cutoff
    plot.appendChild(el("line", { x1: sx(90), x2: sx(90), y1: PAD.t, y2: H - PAD.b,
                                  stroke: "#1f2937", "stroke-width": 1,
                                  "stroke-dasharray": "4 3" }));
    var c = el("text", { x: sx(90) + 5, y: PAD.t + 11, "font-size": 10,
                         fill: "#1f2937", "font-family": "Figtree, sans-serif" });
    c.textContent = "TPSA 90";
    plot.appendChild(c);

    var xt = el("text", { x: PAD.l + (W - PAD.l - PAD.r) / 2, y: H - 8,
                          "text-anchor": "middle", "font-size": 11, fill: "#1f2937",
                          "font-family": "Figtree, sans-serif" });
    xt.textContent = "polar surface area";
    plot.appendChild(xt);

    var ym = PAD.t + (H - PAD.t - PAD.b) / 2;
    var yt = el("text", { x: 12, y: ym, "text-anchor": "middle", "font-size": 11,
                          fill: "#1f2937", "font-family": "Figtree, sans-serif",
                          transform: "rotate(-90 12 " + ym + ")" });
    yt.textContent = "logP";
    plot.appendChild(yt);
  }

  var marker = null;
  function placeMarker(f) {
    if (marker) marker.remove();
    marker = el("g", { transform: "translate(" + sx(f.tpsa) + "," + sy(f.logp) +
                       ") rotate(45)" });
    marker.appendChild(el("rect", { x: -6, y: -6, width: 12, height: 12, rx: 2,
                                    fill: "#fff", stroke: "#1f2937",
                                    "stroke-width": 2.2 }));
    plot.appendChild(marker);
  }

  /* --------------------------------------------------------------- update */

  function fail(message, keepDrawing) {
    alertBox.textContent = message;
    alertBox.hidden = false;
    word.textContent = "—";
    num.textContent = "—";
    fill.style.width = "0%";
    whyBody.innerHTML = "";
    $("rule-checks").innerHTML = "";
    $("rule-note").textContent = "";
    if (!keepDrawing) { drawing.innerHTML = ""; formula.textContent = ""; }
    if (marker) { marker.remove(); marker = null; }
  }

  function update() {
    if (!RDKit) return;
    var smiles = box.value.trim();
    if (!smiles) return fail("Enter a SMILES string to see a prediction.");

    var mol = null;
    try {
      mol = RDKit.get_mol(smiles);
    } catch (e) {
      mol = null;
    }
    if (!mol || !mol.is_valid()) {
      if (mol) mol.delete();
      return fail("RDKit could not read that as a molecule. Check the SMILES.");
    }

    try {
      alertBox.hidden = true;

      drawing.innerHTML = mol.get_svg(330, 210);
      var desc = JSON.parse(mol.get_descriptors());
      formula.textContent = desc.NumHeavyAtoms + " heavy atoms";

      var out = P.score(desc, M);
      var pct = Math.round(out.probability * 100);

      word.textContent = verdictWord(out.probability);
      num.textContent = out.probability.toFixed(2);
      fill.style.width = pct + "%";
      fill.style.background = out.probability >= 0.5 ? "#1d4ed8" : "#b45309";

      renderWhy(out.parts);
      renderRule(out.rule);
      placeMarker(out.features);
    } finally {
      mol.delete();
    }
  }

  /* ----------------------------------------------------------------- init */

  function renderScores() {
    var m = M.metrics;
    // these only exist if the page shows the random-vs-scaffold comparison
    if ($("random-auc")) $("random-auc").textContent = m.random_auc.toFixed(3);
    if ($("scaffold-auc")) $("scaffold-auc").textContent = m.auc.toFixed(3);
    if ($("gap")) $("gap").textContent = m.gap.toFixed(3) + " ROC-AUC";
    if (!$("scores")) return;
    $("scores").innerHTML = [
      [m.auc.toFixed(3), "ROC-AUC, scaffold split"],
      [m.tpsa_auc.toFixed(3), "polar surface area alone"],
      [m.rule_auc.toFixed(3), "CNS rule of thumb"],
      [m.n_total, "measured molecules"],
      [m.n_scaffolds, "distinct scaffolds"]
    ].map(function (s) {
      return '<div class="score"><span class="score__num">' + s[0] +
             '</span><span class="score__lab">' + s[1] + "</span></div>";
    }).join("");
  }

  drawPlotBase();
  renderScores();

  box.addEventListener("input", update);
  $("examples").addEventListener("click", function (e) {
    var btn = e.target.closest("button[data-smiles]");
    if (!btn) return;
    box.value = btn.dataset.smiles;
    update();
  });

  // RDKit_minimal.js exposes initRDKitModule; the wasm sits beside it
  initRDKitModule({ locateFile: function (f) { return "vendor/" + f; } })
    .then(function (instance) {
      RDKit = instance;
      update();
    })
    .catch(function () {
      drawing.innerHTML = "";
      fail("The chemistry engine failed to load, so nothing can be computed. " +
           "Reload the page to try again.", true);
    });
})();
