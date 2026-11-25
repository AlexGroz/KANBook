/* ==========================================================================
   KANBook — общая логика книги:
   тема, боковое оглавление, прогресс чтения, навигация, копирование кода.
   Без внешних библиотек.
   ========================================================================== */
(function () {
  "use strict";

  var root = document.documentElement;
  var THEME_KEY = "kanbook-theme";

  /* ---------- Тема ---------- */

  function currentTheme() {
    return root.getAttribute("data-theme") === "dark" ? "dark" : "light";
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch (e) {
      /* приватный режим — просто игнорируем */
    }
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) {
      meta.setAttribute("content", theme === "dark" ? "#12141a" : "#f7f5ef");
    }
    var btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.setAttribute(
        "aria-label",
        theme === "dark" ? "Включить светлую тему" : "Включить тёмную тему"
      );
      btn.setAttribute("title", theme === "dark" ? "Светлая тема (T)" : "Тёмная тема (T)");
    }
  }

  function toggleTheme() {
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
  }

  /* ---------- Боковое оглавление ---------- */

  function buildToc() {
    var tocHost = document.getElementById("sidebar-toc");
    var article = document.querySelector(".content");
    if (!tocHost || !article) return [];

    var headings = article.querySelectorAll("h2[id], h3[id]");
    if (!headings.length) return [];

    var list = document.createElement("ul");
    list.className = "toc";

    Array.prototype.forEach.call(headings, function (heading) {
      var li = document.createElement("li");
      li.className = heading.tagName === "H2" ? "toc-h2" : "toc-h3";

      var link = document.createElement("a");
      link.href = "#" + heading.id;
      link.textContent = heading.textContent.replace(/\s+/g, " ").trim();
      link.dataset.target = heading.id;
      li.appendChild(link);
      list.appendChild(li);
    });

    tocHost.appendChild(list);
    return Array.prototype.slice.call(list.querySelectorAll("a"));
  }

  function initScrollSpy(links) {
    if (!links.length) return;

    function activate(id) {
      links.forEach(function (link) {
        link.classList.toggle("is-active", link.dataset.target === id);
      });
    }

    if ("IntersectionObserver" in window) {
      var visible = new Set();
      var observer = new IntersectionObserver(
        function (entries) {
          entries.forEach(function (entry) {
            if (entry.isIntersecting) {
              visible.add(entry.target.id);
            } else {
              visible.delete(entry.target.id);
            }
          });
          var first = null;
          var headings = document.querySelectorAll(".content h2[id], .content h3[id]");
          for (var i = 0; i < headings.length; i++) {
            if (visible.has(headings[i].id)) {
              first = headings[i].id;
              break;
            }
          }
          if (first) activate(first);
        },
        { rootMargin: "-72px 0px -68% 0px", threshold: 0 }
      );
      document.querySelectorAll(".content h2[id], .content h3[id]").forEach(function (h) {
        observer.observe(h);
      });
    } else {
      var onScroll = function () {
        var current = null;
        document.querySelectorAll(".content h2[id], .content h3[id]").forEach(function (h) {
          if (h.getBoundingClientRect().top < 140) current = h.id;
        });
        if (current) activate(current);
      };
      window.addEventListener("scroll", onScroll, { passive: true });
      onScroll();
    }
  }

  /* ---------- Прогресс чтения ---------- */

  function initProgress() {
    var bar = document.getElementById("progress-bar");
    if (!bar) return;
    var ticking = false;

    function update() {
      ticking = false;
      var doc = document.documentElement;
      var max = doc.scrollHeight - window.innerHeight;
      var ratio = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
      bar.style.width = (ratio * 100).toFixed(2) + "%";
    }

    window.addEventListener(
      "scroll",
      function () {
        if (!ticking) {
          ticking = true;
          window.requestAnimationFrame(update);
        }
      },
      { passive: true }
    );
    window.addEventListener("resize", update, { passive: true });
    update();
  }

  /* ---------- Мобильное меню ---------- */

  function initSidebar() {
    var toggle = document.getElementById("menu-toggle");
    var scrim = document.getElementById("sidebar-scrim");
    if (!toggle) return;

    function close() {
      document.body.classList.remove("sidebar-open");
      toggle.setAttribute("aria-expanded", "false");
    }

    toggle.addEventListener("click", function () {
      var open = document.body.classList.toggle("sidebar-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });

    if (scrim) scrim.addEventListener("click", close);

    document.querySelectorAll(".sidebar a").forEach(function (a) {
      a.addEventListener("click", function () {
        if (window.innerWidth <= 1080) close();
      });
    });

    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") close();
    });
  }

  /* ---------- Навигация по главам ---------- */

  function chapterHref(selector) {
    var link = document.querySelector(selector);
    if (!link || link.classList.contains("is-disabled")) return null;
    return link.getAttribute("href");
  }

  function initKeyboardNav() {
    document.addEventListener("keydown", function (e) {
      var tag = (e.target && e.target.tagName ? e.target.tagName : "").toLowerCase();
      if (
        tag === "input" ||
        tag === "textarea" ||
        tag === "select" ||
        (e.target && e.target.isContentEditable)
      ) {
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === "ArrowLeft") {
        var prev = chapterHref(".chapter-nav a.prev");
        if (prev) window.location.href = prev;
      } else if (e.key === "ArrowRight") {
        var next = chapterHref(".chapter-nav a.next");
        if (next) window.location.href = next;
      } else if (e.key === "t" || e.key === "T") {
        toggleTheme();
      } else if (e.key === "h" || e.key === "H") {
        var home = document.querySelector(".brand");
        if (home) window.location.href = home.getAttribute("href");
      }
    });
  }

  /* ---------- Копирование кода ---------- */

  function initCopy() {
    document.querySelectorAll(".copy-btn").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var block = btn.closest(".code-block");
        var code = block ? block.querySelector("code") : null;
        if (!code) return;
        var text = code.textContent;
        var done = function () {
          btn.textContent = "Скопировано";
          btn.classList.add("is-done");
          window.setTimeout(function () {
            btn.textContent = "Копировать";
            btn.classList.remove("is-done");
          }, 1600);
        };

        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(done, function () {
            fallbackCopy(text, done);
          });
        } else {
          fallbackCopy(text, done);
        }
      });
    });
  }

  function fallbackCopy(text, done) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-9999px";
    document.body.appendChild(area);
    area.select();
    try {
      document.execCommand("copy");
      done();
    } catch (e) {
      /* ничего не поделать */
    }
    document.body.removeChild(area);
  }

  /* ---------- Подсветка кода (Python) ---------- */

  var KEYWORDS = {
    def: 1, "class": 1, return: 1, import: 1, from: 1, as: 1, "if": 1, elif: 1,
    "else": 1, "for": 1, "while": 1, "in": 1, not: 1, and: 1, or: 1, is: 1,
    None: 1, True: 1, False: 1, lambda: 1, with: 1, try: 1, except: 1,
    finally: 1, raise: 1, yield: 1, global: 1, nonlocal: 1, pass: 1, break: 1,
    continue: 1, assert: 1, del: 1, async: 1, await: 1
  };

  var BUILTINS = {
    print: 1, len: 1, range: 1, sum: 1, abs: 1, min: 1, max: 1, open: 1,
    enumerate: 1, zip: 1, float: 1, int: 1, list: 1, dict: 1, set: 1,
    tuple: 1, super: 1, isinstance: 1, getattr: 1, setattr: 1, hasattr: 1,
    type: 1, round: 1, sorted: 1, map: 1, filter: 1, repr: 1, str: 1,
    bool: 1, format: 1, input: 1
  };

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function highlightPython(source) {
    var re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(@[A-Za-z_]\w*)|\b(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b|\b([A-Za-z_]\w*)\b/g;
    var out = "";
    var last = 0;
    var match;

    while ((match = re.exec(source)) !== null) {
      out += escapeHtml(source.slice(last, match.index));
      var token = match[0];
      var cls = "";

      if (match[1]) {
        cls = "tok-com";
      } else if (match[2]) {
        cls = "tok-str";
      } else if (match[3]) {
        cls = "tok-dec";
      } else if (match[4]) {
        cls = "tok-num";
      } else if (match[5]) {
        var word = match[5];
        var after = source.slice(re.lastIndex, re.lastIndex + 4);
        if (KEYWORDS[word]) {
          cls = "tok-kw";
        } else if (word === "self") {
          cls = "tok-self";
        } else if (BUILTINS[word]) {
          cls = "tok-bi";
        } else if (/^\s*\(/.test(after)) {
          cls = "tok-fn";
        }
      }

      out += cls
        ? '<span class="' + cls + '">' + escapeHtml(token) + "</span>"
        : escapeHtml(token);
      last = re.lastIndex;
    }

    out += escapeHtml(source.slice(last));
    return out;
  }

  function initHighlight() {
    document.querySelectorAll("pre code.language-python").forEach(function (code) {
      if (code.dataset.highlighted === "1") return;
      code.innerHTML = highlightPython(code.textContent);
      code.dataset.highlighted = "1";
    });
  }

  /* ---------- Запуск ---------- */

  function init() {
    var toggle = document.getElementById("theme-toggle");
    if (toggle) toggle.addEventListener("click", toggleTheme);

    applyTheme(currentTheme());
    initSidebar();

    var links = buildToc();
    initScrollSpy(links);

    initProgress();
    initKeyboardNav();
    initCopy();
    initHighlight();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();


