/* ==========================================================================
   Живое демо: обученная KAN 4 -> 3 на Iris.
   Инференс полностью в браузере, без сервера и без внешних библиотек.
   ========================================================================== */
(function () {
  "use strict";

  var M = window.IRIS_KAN_MODEL;
  var form = document.getElementById("iris-form");
  var probsHost = document.getElementById("probs");
  var badge = document.getElementById("result-badge");
  var statsHost = document.getElementById("model-stats");

  if (!M || !form || !probsHost || !badge) {
    if (badge) badge.textContent = "Не удалось загрузить модель (iris_model.js).";
    return;
  }

  var LABELS = {
    sepal_length: "Длина чашелистика, см",
    sepal_width: "Ширина чашелистика, см",
    petal_length: "Длина лепестка, см",
    petal_width: "Ширина лепестка, см"
  };

  var PRESETS = {
    setosa: [5.0, 3.4, 1.5, 0.2],
    versicolor: [5.9, 2.8, 4.3, 1.3],
    virginica: [6.5, 3.0, 5.5, 1.8]
  };

  var inputs = [];

  /* ---------- B-сплайны (те же формулы, что в книге) ---------- */

  function basisRow(knots, degree, x) {
    var m = knots.length;
    var B = new Array(m - 1);
    var i;
    for (i = 0; i < m - 1; i++) {
      B[i] = x >= knots[i] && x < knots[i + 1] ? 1 : 0;
    }
    if (Math.abs(x - knots[m - 1]) < 1e-12) {
      B[m - degree - 2] = 1;
    }
    for (var d = 1; d <= degree; d++) {
      var count = m - d - 1;
      var next = new Array(count);
      for (var j = 0; j < count; j++) {
        var den1 = knots[j + d] - knots[j];
        var den2 = knots[j + d + 1] - knots[j + 1];
        var t1 = den1 > 1e-12 ? ((x - knots[j]) / den1) * B[j] : 0;
        var t2 = den2 > 1e-12 ? ((knots[j + d + 1] - x) / den2) * B[j + 1] : 0;
        next[j] = t1 + t2;
      }
      B = next;
    }
    return B;
  }

  function silu(x) {
    return x / (1 + Math.exp(-x));
  }

  function softmax(z) {
    var m = Math.max.apply(null, z);
    var e = z.map(function (v) {
      return Math.exp(v - m);
    });
    var s = e.reduce(function (a, b) {
      return a + b;
    }, 0);
    return e.map(function (v) {
      return v / s;
    });
  }

  /* ---------- Инференс ---------- */

  function normalize(value, i) {
    var lo = M.min[i];
    var hi = M.max[i];
    var x = hi > lo ? -1 + (2 * (value - lo)) / (hi - lo) : 0;
    return Math.max(-1, Math.min(1, x));
  }

  function predict(values) {
    var xn = values.map(normalize);
    var basis = xn.map(function (x) {
      return basisRow(M.knots, M.degree, x);
    });

    var logits = [];
    for (var k = 0; k < M.classes.length; k++) {
      var z = M.bias[k];
      for (var i = 0; i < M.features.length; i++) {
        z += M.base[k][i] * silu(xn[i]);
        var coef = M.coef[k][i];
        for (var b = 0; b < coef.length; b++) {
          z += coef[b] * basis[i][b];
        }
      }
      logits.push(z);
    }
    return softmax(logits);
  }
  /* ---------- Интерфейс ---------- */

  function readValues() {
    return inputs.map(function (input) {
      return parseFloat(input.value);
    });
  }

  function render() {
    var values = readValues();
    var probs = predict(values);
    var top = 0;
    for (var k = 1; k < probs.length; k++) {
      if (probs[k] > probs[top]) top = k;
    }

    badge.innerHTML =
      'Похоже на <strong>' + M.classes[top] + "</strong>" +
      "<span>уверенность " + (probs[top] * 100).toFixed(4) + "%</span>";

    probsHost.innerHTML = M.classes.map(function (name, k) {
      var pct = (probs[k] * 100).toFixed(4);
      return (
        '<div class="prob-row">' +
          '<div class="prob-head"><span>' + name + "</span><strong>" + pct + "%</strong></div>" +
          '<div class="prob-track"><div class="prob-fill" style="width:' + pct + '%"></div></div>' +
        "</div>"
      );
    }).join("");
  }

  function setValues(values) {
    inputs.forEach(function (input, i) {
      var lo = parseFloat(input.min);
      var hi = parseFloat(input.max);
      var value = Math.max(lo, Math.min(hi, values[i]));
      input.value = value;
      var out = document.getElementById("out-" + i);
      if (out) out.textContent = value.toFixed(1);
    });
    render();
  }

  function buildFields() {
    M.features.forEach(function (name, i) {
      var min = M.min[i];
      var max = M.max[i];
      var row = document.createElement("div");
      row.className = "field-row";

      var label = document.createElement("label");
      label.setAttribute("for", "f" + i);
      label.textContent = LABELS[name] || name;

      var out = document.createElement("output");
      out.id = "out-" + i;
      out.setAttribute("for", "f" + i);

      var input = document.createElement("input");
      input.type = "range";
      input.id = "f" + i;
      input.min = min;
      input.max = max;
      input.step = "0.1";
      input.value = (min + max) / 2;
      input.addEventListener("input", function () {
        out.textContent = parseFloat(input.value).toFixed(1);
        render();
      });

      row.appendChild(label);
      row.appendChild(out);
      row.appendChild(input);
      form.appendChild(row);
      inputs.push(input);
    });
  }

  function buildPresets() {
    var host = document.getElementById("presets");
    if (!host) return;
    Object.keys(PRESETS).forEach(function (name) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-sm";
      btn.textContent = name;
      btn.addEventListener("click", function () {
        setValues(PRESETS[name]);
      });
      host.appendChild(btn);
    });
  }

  function buildStats() {
    if (!statsHost) return;
    var nParams = M.classes.length * M.features.length * (M.grid + M.degree + 1) + M.classes.length;
    var testAcc = M.metrics && M.metrics.test_accuracy != null
      ? (M.metrics.test_accuracy * 100).toFixed(0) + "%"
      : "—";
    var testN = M.metrics && M.metrics.test_n != null ? M.metrics.test_n : "—";

    var cells = [
      ["Параметров", String(nParams)],
      ["Тест", testAcc + " (" + testN + ")"],
      ["Признаков", String(M.features.length)],
      ["Классов", String(M.classes.length)]
    ];

    statsHost.innerHTML = cells.map(function (pair) {
      return "<div><dt>" + pair[0] + "</dt><dd>" + pair[1] + "</dd></div>";
    }).join("");
  }

  buildFields();
  buildPresets();
  buildStats();
  setValues(PRESETS.versicolor);
})();

