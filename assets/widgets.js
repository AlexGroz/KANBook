/* ==========================================================================
   KANBook — интерактивные демонстрации.
   Каждый блок активируется только если на странице есть его разметка.
   ========================================================================== */
(function () {
  "use strict";

  var COLORS = {
    bg: "#10131a",
    grid: "rgba(255,255,255,0.07)",
    axis: "rgba(255,255,255,0.22)",
    text: "#8f97a6",
    accent: "#f0a05a",
    teal: "#5ec7b8",
    paper: "rgba(255,255,255,0.82)"
  };

  function setupCanvas(canvas, width, height) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.aspectRatio = width + " / " + height;
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  /* ======================================================================
     1. Градиентный спуск на двумерной функции
     ====================================================================== */

  function initGradientDemo() {
    var host = document.querySelector('[data-demo="gd"]');
    if (!host) return;

    var canvas = host.querySelector("canvas");
    var lrInput = host.querySelector('input[type="range"]');
    var lrLabel = host.querySelector("[data-lr-value]");
    var out = host.querySelector(".demo-output");
    var stepBtn = host.querySelector('[data-action="step"]');
    var runBtn = host.querySelector('[data-action="run"]');
    var resetBtn = host.querySelector('[data-action="reset"]');
    if (!canvas || !lrInput || !out) return;

    var W = 640;
    var H = 380;
    var ctx = setupCanvas(canvas, W, H);
    var domain = 2.6;
    var path = [];
    var point = { x: 2.15, y: 1.75 };
    var timer = null;

    function f(x, y) {
      return Math.sin(x) * Math.cos(y) + 0.08 * (x * x + y * y);
    }

    function grad(x, y) {
      return {
        x: Math.cos(x) * Math.cos(y) + 0.16 * x,
        y: -Math.sin(x) * Math.sin(y) + 0.16 * y
      };
    }

    function toScreen(x, y) {
      return {
        x: ((x + domain) / (2 * domain)) * W,
        y: ((domain - y) / (2 * domain)) * H
      };
    }

    function toWorld(px, py) {
      return {
        x: (px / W) * 2 * domain - domain,
        y: domain - (py / H) * 2 * domain
      };
    }

    /* Фон с изолиниями — считаем один раз в offscreen-канве. */
    var background = document.createElement("canvas");
    background.width = 320;
    background.height = 190;
    var bgCtx = background.getContext("2d");
    var image = bgCtx.createImageData(320, 190);
    var vMin = -1.35;
    var vMax = 1.6;

    for (var py = 0; py < 190; py++) {
      for (var px = 0; px < 320; px++) {
        var wx = (px / 320) * 2 * domain - domain;
        var wy = domain - (py / 190) * 2 * domain;
        var v = f(wx, wy);
        var t = clamp((v - vMin) / (vMax - vMin), 0, 1);
        var r = 12 + 150 * t * t;
        var g = 20 + 96 * t;
        var b = 36 + 52 * t * t;
        var band = Math.abs((v * 2.2) % 1);
        if (band < 0.045 || band > 0.955) {
          r *= 0.72;
          g *= 0.72;
          b *= 0.72;
        }
        var idx = (py * 320 + px) * 4;
        image.data[idx] = r;
        image.data[idx + 1] = g;
        image.data[idx + 2] = b;
        image.data[idx + 3] = 255;
      }
    }
    bgCtx.putImageData(image, 0, 0);

    function drawBackground() {
      ctx.clearRect(0, 0, W, H);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(background, 0, 0, W, H);
    }

    function drawPath() {
      if (path.length > 0) {
        ctx.beginPath();
        path.forEach(function (p, i) {
          var s = toScreen(p.x, p.y);
          if (i === 0) ctx.moveTo(s.x, s.y);
          else ctx.lineTo(s.x, s.y);
        });
        ctx.strokeStyle = "rgba(240,160,90,0.95)";
        ctx.lineWidth = 2;
        ctx.lineJoin = "round";
        ctx.stroke();

        path.forEach(function (p, i) {
          if (i === 0 || i === path.length - 1) return;
          var s = toScreen(p.x, p.y);
          ctx.beginPath();
          ctx.arc(s.x, s.y, 2.2, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(240,160,90,0.6)";
          ctx.fill();
        });
      }

      var start = toScreen(point.x, point.y);
      ctx.beginPath();
      ctx.arc(start.x, start.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.teal;
      ctx.fill();
      ctx.strokeStyle = "#10131a";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    function report() {
      var g = grad(point.x, point.y);
      var norm = Math.sqrt(g.x * g.x + g.y * g.y);
      out.textContent =
        "шаг " + path.length +
        " · x = (" + point.x.toFixed(2) + ", " + point.y.toFixed(2) + ")" +
        " · f = " + f(point.x, point.y).toFixed(3) +
        " · ‖∇f‖ = " + norm.toFixed(3);
    }

    function draw() {
      drawBackground();
      drawPath();
      report();
    }

    function step() {
      var g = grad(point.x, point.y);
      var lr = parseFloat(lrInput.value);
      var nx = point.x - lr * g.x;
      var ny = point.y - lr * g.y;
      if (!isFinite(nx) || !isFinite(ny)) return false;
      point = {
        x: clamp(nx, -domain, domain),
        y: clamp(ny, -domain, domain)
      };
      path.push({ x: point.x, y: point.y });
      draw();
      var norm = Math.sqrt(g.x * g.x + g.y * g.y);
      return norm > 0.004 && path.length < 400;
    }

    function stopAuto() {
      if (timer) {
        window.clearInterval(timer);
        timer = null;
      }
      if (runBtn) runBtn.textContent = "Авто";
    }

    function reset(x, y) {
      stopAuto();
      point = { x: x, y: y };
      path = [{ x: x, y: y }];
      draw();
    }

    lrInput.addEventListener("input", function () {
      if (lrLabel) lrLabel.textContent = parseFloat(lrInput.value).toFixed(2);
    });

    if (stepBtn) {
      stepBtn.addEventListener("click", function () {
        stopAuto();
        if (path.length === 0) path.push({ x: point.x, y: point.y });
        step();
      });
    }

    if (runBtn) {
      runBtn.addEventListener("click", function () {
        if (timer) {
          stopAuto();
          return;
        }
        runBtn.textContent = "Пауза";
        timer = window.setInterval(function () {
          if (!step()) stopAuto();
        }, 110);
      });
    }

    if (resetBtn) {
      resetBtn.addEventListener("click", function () {
        reset(2.15, 1.75);
      });
    }

    canvas.addEventListener("click", function (event) {
      var rect = canvas.getBoundingClientRect();
      var px = ((event.clientX - rect.left) / rect.width) * W;
      var py = ((event.clientY - rect.top) / rect.height) * H;
      var world = toWorld(px, py);
      reset(world.x, world.y);
    });

    reset(2.15, 1.75);
  }

  /* ======================================================================
     2. Графики функций активации
     ====================================================================== */

  function initActivationDemo() {
    var host = document.querySelector('[data-demo="activation"]');
    if (!host) return;

    var canvas = host.querySelector("canvas");
    var out = host.querySelector(".demo-output");
    var buttons = host.querySelectorAll("[data-fn]");
    if (!canvas) return;

    var W = 640;
    var H = 280;
    var ctx = setupCanvas(canvas, W, H);
    var xMin = -6;
    var xMax = 6;
    var yMin = -1.6;
    var yMax = 1.6;
    var current = "silu";

    var FUNCS = {
      sigmoid: {
        label: "σ(x) = 1 / (1 + e⁻ˣ)",
        fn: function (x) {
          return 1 / (1 + Math.exp(-x));
        }
      },
      tanh: {
        label: "tanh(x)",
        fn: function (x) {
          return Math.tanh(x);
        }
      },
      relu: {
        label: "ReLU(x) = max(0, x)",
        fn: function (x) {
          return Math.max(0, x);
        }
      },
      leaky: {
        label: "LeakyReLU(x) = max(0.1x, x)",
        fn: function (x) {
          return x >= 0 ? x : 0.1 * x;
        }
      },
      gelu: {
        label: "GELU ≈ 0.5x(1 + tanh(√(2/π)(x + 0.0447x³)))",
        fn: function (x) {
          var c = Math.sqrt(2 / Math.PI);
          return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
        }
      },
      silu: {
        label: "SiLU(x) = x / (1 + e⁻ˣ)",
        fn: function (x) {
          return x / (1 + Math.exp(-x));
        }
      }
    };

    function mapX(x) {
      return ((x - xMin) / (xMax - xMin)) * W;
    }

    function mapY(y) {
      var pad = 18;
      return H - pad - ((y - yMin) / (yMax - yMin)) * (H - 2 * pad);
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = COLORS.bg;
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      for (var gx = -6; gx <= 6; gx += 1) {
        var sx = mapX(gx);
        ctx.beginPath();
        ctx.moveTo(sx, 0);
        ctx.lineTo(sx, H);
        ctx.stroke();
      }
      for (var gy = -1.5; gy <= 1.5; gy += 0.5) {
        var sy = mapY(gy);
        ctx.beginPath();
        ctx.moveTo(0, sy);
        ctx.lineTo(W, sy);
        ctx.stroke();
      }

      ctx.strokeStyle = COLORS.axis;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(0, mapY(0));
      ctx.lineTo(W, mapY(0));
      ctx.moveTo(mapX(0), 0);
      ctx.lineTo(mapX(0), H);
      ctx.stroke();

      ctx.fillStyle = COLORS.text;
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("−6", mapX(-6) + 4, mapY(0) + 16);
      ctx.fillText("6", mapX(6) - 10, mapY(0) + 16);
      ctx.fillText("0", mapX(0) + 6, mapY(0) + 16);

      var fn = FUNCS[current].fn;
      ctx.beginPath();
      var started = false;
      for (var px = 0; px <= W; px += 1.5) {
        var x = xMin + (px / W) * (xMax - xMin);
        var y = fn(x);
        var py = mapY(clamp(y, yMin, yMax));
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2.4;
      ctx.lineJoin = "round";
      ctx.stroke();

      if (out) out.textContent = FUNCS[current].label;
    }

    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        current = btn.dataset.fn;
        buttons.forEach(function (b) {
          b.setAttribute("aria-pressed", b === btn ? "true" : "false");
        });
        draw();
      });
    });

    draw();
  }

  /* ======================================================================
     3. B-сплайны и аппроксимация одномерной функции
     ====================================================================== */

  /* Равномерные «зажатые» узлы: k+1 копия на концах, G−1 внутри.
     Число базисных функций = G + k. */
  function buildKnots(G, k) {
    var knots = [];
    var i;
    for (i = 0; i <= k; i++) knots.push(-1);
    for (i = 1; i < G; i++) knots.push(-1 + (2 * i) / G);
    for (i = 0; i <= k; i++) knots.push(1);
    return knots;
  }

  function basisRow(knots, k, x) {
    var m = knots.length;
    var B = new Float64Array(m - 1);
    var i;
    for (i = 0; i < m - 1; i++) {
      B[i] = x >= knots[i] && x < knots[i + 1] ? 1 : 0;
    }
    var lastNonEmpty = m - k - 2;
    if (Math.abs(x - knots[m - 1]) < 1e-12) {
      B[lastNonEmpty] = 1;
    }
    for (var d = 1; d <= k; d++) {
      var count = m - d - 1;
      var next = new Float64Array(count);
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

  /* Решение нормальных уравнений M c = r методом Гаусса. */
  function solveNormalEquations(M, r) {
    var n = r.length;
    var a = [];
    var i;
    var j;
    for (i = 0; i < n; i++) a.push(Array.prototype.slice.call(M[i]).concat([r[i]]));

    for (var col = 0; col < n; col++) {
      var pivot = col;
      for (i = col + 1; i < n; i++) {
        if (Math.abs(a[i][col]) > Math.abs(a[pivot][col])) pivot = i;
      }
      var tmp = a[col];
      a[col] = a[pivot];
      a[pivot] = tmp;

      var div = a[col][col];
      if (Math.abs(div) < 1e-14) continue;

      for (j = col; j <= n; j++) a[col][j] /= div;
      for (i = 0; i < n; i++) {
        if (i === col) continue;
        var factor = a[i][col];
        if (Math.abs(factor) < 1e-15) continue;
        for (j = col; j <= n; j++) a[i][j] -= factor * a[col][j];
      }
    }

    var c = new Float64Array(n);
    for (i = 0; i < n; i++) c[i] = a[i][n];
    return c;
  }

  function initSplineDemo() {
    var host = document.querySelector('[data-demo="spline"]');
    if (!host) return;

    var canvas = host.querySelector("canvas");
    var degreeSelect = host.querySelector("[data-spline-degree]");
    var gridInput = host.querySelector("[data-spline-grid]");
    var gridLabel = host.querySelector("[data-grid-value]");
    var targetBtn = host.querySelector('[data-action="target"]');
    var out = host.querySelector(".demo-output");
    if (!canvas) return;

    var W = 700;
    var H = 320;
    var ctx = setupCanvas(canvas, W, H);
    var pad = 26;
    var yRange = 1.5;
    var samples = 200;

    var targets = [
      {
        label: "sin(2.2x) + 0.35x",
        fn: function (x) {
          return Math.sin(2.2 * x) + 0.35 * x;
        }
      },
      {
        label: "0.9·cos(3x) − 0.2",
        fn: function (x) {
          return 0.9 * Math.cos(3 * x) - 0.2;
        }
      },
      {
        label: "tanh(4x)·0.8",
        fn: function (x) {
          return 0.8 * Math.tanh(4 * x);
        }
      },
      {
        label: "x³ − x",
        fn: function (x) {
          return x * x * x - x;
        }
      }
    ];
    var targetIndex = 0;

    function mapX(x) {
      return pad + ((x + 1) / 2) * (W - 2 * pad);
    }

    function mapY(y) {
      return H / 2 - (y / yRange) * (H / 2 - pad);
    }

    function fit() {
      var k = parseInt(degreeSelect ? degreeSelect.value : "3", 10);
      var G = parseInt(gridInput ? gridInput.value : "5", 10);
      var knots = buildKnots(G, k);
      var n = G + k;
      var rows = [];
      var ys = [];
      var i;
      var j;

      for (i = 0; i < samples; i++) {
        var x = -1 + (2 * i) / (samples - 1);
        rows.push(basisRow(knots, k, x));
        ys.push(targets[targetIndex].fn(x));
      }

      var M = [];
      var r = new Array(n).fill(0);
      for (i = 0; i < n; i++) M.push(new Float64Array(n));

      for (var s = 0; s < samples; s++) {
        var row = rows[s];
        for (i = 0; i < n; i++) {
          r[i] += row[i] * ys[s];
          for (j = 0; j < n; j++) M[i][j] += row[i] * row[j];
        }
      }
      for (i = 0; i < n; i++) M[i][i] += 1e-8;

      return {
        k: k,
        G: G,
        knots: knots,
        n: n,
        coef: solveNormalEquations(M, r)
      };
    }

    function splineValue(model, x) {
      var B = basisRow(model.knots, model.k, x);
      var sum = 0;
      for (var i = 0; i < model.n; i++) sum += model.coef[i] * B[i];
      return sum;
    }

    function draw() {
      var model = fit();
      var target = targets[targetIndex];
      var px;
      var i;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = COLORS.bg;
      ctx.fillRect(0, 0, W, H);

      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      for (i = 0; i < model.knots.length; i++) {
        var kx = mapX(model.knots[i]);
        ctx.beginPath();
        ctx.moveTo(kx, pad - 8);
        ctx.lineTo(kx, H - pad + 8);
        ctx.stroke();
      }

      ctx.strokeStyle = COLORS.axis;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.moveTo(pad - 10, mapY(0));
      ctx.lineTo(W - pad + 10, mapY(0));
      ctx.moveTo(mapX(0), pad - 10);
      ctx.lineTo(mapX(0), H - pad + 10);
      ctx.stroke();

      ctx.fillStyle = COLORS.text;
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("−1", mapX(-1) - 6, H - 8);
      ctx.fillText("1", mapX(1) - 3, H - 8);
      ctx.fillText("x", W - pad + 8, mapY(0) + 4);

      /* Целевая функция */
      ctx.beginPath();
      for (px = pad; px <= W - pad; px += 2) {
        var tx = -1 + ((px - pad) / (W - 2 * pad)) * 2;
        var ty = target.fn(tx);
        if (px === pad) ctx.moveTo(px, mapY(ty));
        else ctx.lineTo(px, mapY(ty));
      }
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = "rgba(255,255,255,0.65)";
      ctx.lineWidth = 1.6;
      ctx.stroke();
      ctx.setLineDash([]);

      /* Базисные функции */
      for (i = 0; i < model.n; i++) {
        var hue = model.n > 1 ? 24 + (i / (model.n - 1)) * 152 : 40;
        ctx.beginPath();
        for (px = pad; px <= W - pad; px += 3) {
          var bx = -1 + ((px - pad) / (W - 2 * pad)) * 2;
          var B = basisRow(model.knots, model.k, bx);
          var by = mapY(B[i]);
          if (px === pad) ctx.moveTo(px, by);
          else ctx.lineTo(px, by);
        }
        ctx.strokeStyle = "hsla(" + hue.toFixed(0) + ", 65%, 62%, 0.55)";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }

      /* Сумма базисных функций с коэффициентами — это и есть «сплайн» */
      ctx.beginPath();
      for (px = pad; px <= W - pad; px += 2) {
        var sx = -1 + ((px - pad) / (W - 2 * pad)) * 2;
        var sy = mapY(splineValue(model, sx));
        if (px === pad) ctx.moveTo(px, sy);
        else ctx.lineTo(px, sy);
      }
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2.6;
      ctx.lineJoin = "round";
      ctx.stroke();

      var error = 0;
      for (i = 0; i < samples; i++) {
        var ex = -1 + (2 * i) / (samples - 1);
        var diff = splineValue(model, ex) - target.fn(ex);
        error += diff * diff;
      }
      var rmse = Math.sqrt(error / samples);

      if (out) {
        out.textContent =
          "G = " + model.G +
          " · k = " + model.k +
          " · базисных функций G + k = " + model.n +
          " · RMSE = " + rmse.toFixed(4) +
          " · цель: " + target.label;
      }
    }

    if (gridInput) {
      gridInput.addEventListener("input", function () {
        if (gridLabel) gridLabel.textContent = gridInput.value;
        draw();
      });
    }
    if (degreeSelect) degreeSelect.addEventListener("change", draw);
    if (targetBtn) {
      targetBtn.addEventListener("click", function () {
        targetIndex = (targetIndex + 1) % targets.length;
        draw();
      });
    }

    draw();
  }

  /* ======================================================================
     4. Счётчик параметров: KAN против MLP
     ====================================================================== */

  function initParamDemo() {
    var host = document.querySelector('[data-demo="params"]');
    if (!host) return;

    var inputs = host.querySelector('[data-param="inputs"]');
    var width = host.querySelector('[data-param="width"]');
    var layers = host.querySelector('[data-param="layers"]');
    var grid = host.querySelector('[data-param="grid"]');
    var out = host.querySelector(".demo-output");
    if (!inputs || !width || !layers || !grid || !out) return;

    var k = 3;

    function format(value) {
      return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
    }

    function kanParams(n, m, L, G) {
      var perEdge = G + k + 1;
      var p = n * m * perEdge + m;
      for (var i = 1; i < L; i++) p += m * m * perEdge + m;
      p += m * perEdge + 1;
      return p;
    }

    function mlpParams(n, m, L) {
      var p = n * m + m;
      for (var i = 1; i < L; i++) p += m * m + m;
      p += m + 1;
      return p;
    }

    function update() {
      var n = clamp(parseInt(inputs.value, 10) || 1, 1, 16);
      var m = clamp(parseInt(width.value, 10) || 2, 2, 64);
      var L = clamp(parseInt(layers.value, 10) || 1, 1, 4);
      var G = clamp(parseInt(grid.value, 10) || 2, 2, 16);

      var kan = kanParams(n, m, L, G);
      var mlp = mlpParams(n, m, L);
      var ratio = kan / mlp;

      out.innerHTML =
        "MLP: <strong>" + format(mlp) + "</strong> параметров" +
        " · KAN: <strong>" + format(kan) + "</strong> параметров" +
        " · отношение ≈ <strong>" + ratio.toFixed(1) + "×</strong>" +
        " · на одно ребро KAN: G + k + 1 = " + (G + k + 1) + " параметров";
    }

    [inputs, width, layers, grid].forEach(function (el) {
      el.addEventListener("input", update);
    });

    update();
  }

  /* ---------- Запуск ---------- */

  /* Небольшой публичный API: удобно проверять формулы в консоли браузера. */
  window.KANBook = {
    buildKnots: buildKnots,
    basisRow: basisRow,
    solveNormalEquations: solveNormalEquations
  };

  function init() {
    initGradientDemo();
    initActivationDemo();
    initSplineDemo();
    initParamDemo();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();





