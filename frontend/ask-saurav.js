/*!
 * Ask Saurav — embeddable RAG chat widget (no dependencies, Shadow-DOM isolated).
 *
 * <script src="ask-saurav.js" data-api="https://<your-space>.hf.space" defer></script>
 *
 * Options (data-* on the script tag):
 *   data-api        Backend base URL (default: same origin)
 *   data-greeting   First message shown when the panel opens
 *   data-teaser     "false" to disable the small hint bubble next to the launcher
 * Any element with [data-ask-saurav] opens the chat; data-ask-saurav="question" also asks it.
 * JS API: window.AskSaurav.open(question?), window.AskSaurav.close()
 */
(function () {
  "use strict";
  if (window.AskSaurav) return;

  var script = document.currentScript || document.querySelector("script[src*='ask-saurav']");
  var ds = (script && script.dataset) || {};
  var API = (ds.api || "").replace(/\/+$/, "");
  var GREETING =
    ds.greeting ||
    "Hi, I'm Saurav. Well, the AI version of me, built on my resume, papers and projects. Ask me about my research, internships, skills or how I built this chatbot.";
  var FALLBACK_SUGGESTIONS = [
    "Tell me about yourself",
    "What is AURA?",
    "What did you do at L&T Smart City?",
    "What's your Gen AI / LLM experience?",
    "How did you build this chatbot?",
    "How can I contact you?",
  ];

  // ---------- state ----------
  var history = [];
  var busy = false;
  var serverReady = false;
  var sessionId = (function () {
    try {
      var k = "ask-saurav-sid", v = sessionStorage.getItem(k);
      if (!v) { v = Math.random().toString(36).slice(2) + Date.now().toString(36); sessionStorage.setItem(k, v); }
      return v;
    } catch (e) { return Math.random().toString(36).slice(2); }
  })();

  // ---------- styles ----------
  var CSS = `
  :host{all:initial;--bg:#080520;--bg2:#0C0730;--line:#25206A;--line2:#332E88;--tx:#F0EEFF;--mu:#A3A3D1;
    --cy:#00F5C8;--vi:#9333EA;--pi:#FF2D9B;--or:#FF6B1A;
    --fd:'Syne','DM Sans',system-ui,sans-serif;--fb:'DM Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
    font-family:var(--fb);color:var(--tx);position:fixed;z-index:9000;right:24px;bottom:24px}
  *{box-sizing:border-box;margin:0;padding:0}
  button{font:inherit;color:inherit;cursor:pointer;border:0;background:none}
  :focus-visible{outline:2px solid var(--cy);outline-offset:2px}

  .ring{position:relative;flex:none;border-radius:50%;display:grid;place-items:center}
  .ring::before{content:"";position:absolute;inset:0;border-radius:50%;
    background:conic-gradient(var(--cy),var(--vi) 30%,var(--pi) 60%,var(--or) 85%,var(--cy))}
  .ring::after{content:"";position:absolute;inset:2px;border-radius:50%;background:var(--bg)}
  .ring span{position:relative;z-index:1;font-family:var(--fd);font-weight:800;letter-spacing:-.03em;color:var(--tx)}
  :host(.thinking) .ring::before,.thinking .ring::before{animation:spin 1.1s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}

  .launcher{display:flex;align-items:center;gap:12px;padding:7px 18px 7px 7px;border-radius:999px;
    background:rgba(8,5,32,.92);border:1px solid var(--line2);backdrop-filter:blur(14px);
    box-shadow:0 12px 40px rgba(123,47,190,.35);transition:transform .25s,border-color .25s}
  .launcher:hover{transform:translateY(-3px);border-color:var(--cy)}
  .launcher .ring{width:44px;height:44px}.launcher .ring span{font-size:15px}
  .launcher b{font-family:var(--fd);font-weight:700;font-size:15px;display:block;line-height:1.15}
  .launcher small{display:block;font-size:12px;color:var(--mu);line-height:1.3}
  .launcher[hidden]{display:none}

  .teaser{position:absolute;right:0;bottom:72px;width:250px;padding:14px 34px 14px 16px;border-radius:16px 16px 4px 16px;
    background:var(--bg2);border:1px solid var(--line2);font-size:14px;line-height:1.5;color:var(--tx);
    box-shadow:0 16px 40px rgba(0,0,0,.4);animation:pop .35s cubic-bezier(.34,1.4,.64,1)}
  .teaser .x{position:absolute;top:6px;right:8px;width:24px;height:24px;color:var(--mu);font-size:18px;line-height:1}
  .teaser[hidden]{display:none}
  @keyframes pop{from{opacity:0;transform:translateY(8px) scale(.96)}to{opacity:1;transform:none}}

  .panel{position:absolute;right:0;bottom:0;width:400px;height:min(640px,calc(100vh - 48px));display:flex;flex-direction:column;
    background:var(--bg);border:1px solid var(--line2);border-radius:22px;overflow:hidden;
    box-shadow:0 30px 80px rgba(0,0,0,.55),0 0 0 1px rgba(0,245,200,.04);
    transform-origin:bottom right;animation:open .3s cubic-bezier(.34,1.3,.64,1)}
  .panel[hidden]{display:none}
  @keyframes open{from{opacity:0;transform:scale(.92) translateY(12px)}to{opacity:1;transform:none}}

  header{display:flex;align-items:center;gap:12px;padding:16px 14px 14px 18px;border-bottom:1px solid var(--line);
    background:linear-gradient(180deg,rgba(123,47,190,.16),transparent)}
  header .ring{width:42px;height:42px}header .ring span{font-size:14px}
  .who{flex:1;min-width:0}
  .who h2{font-family:var(--fd);font-size:17px;font-weight:700;letter-spacing:-.01em;line-height:1.2}
  .who p{font-size:12.5px;color:var(--mu);line-height:1.35}
  .dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--cy);margin-right:6px;vertical-align:1px;
    box-shadow:0 0 8px var(--cy)}
  .dot.wait{background:var(--or);box-shadow:0 0 8px var(--or)}
  .iconbtn{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;color:var(--mu);transition:background .2s,color .2s}
  .iconbtn:hover{background:rgba(255,255,255,.06);color:var(--tx)}

  .log{flex:1;overflow-y:auto;padding:20px 18px 8px;display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth;
    scrollbar-width:thin;scrollbar-color:var(--line2) transparent}
  .msg{font-size:14.5px;line-height:1.6;max-width:100%;overflow-wrap:anywhere}
  .me{align-self:flex-end;max-width:85%;padding:9px 14px;border-radius:16px 16px 4px 16px;
    background:linear-gradient(135deg,var(--vi),#7B2FBE);color:#fff}
  .ai{padding-left:14px;border-left:2px solid var(--cy)}
  .ai p+p,.ai p+ul,.ai ul+p{margin-top:8px}
  .ai ul{padding-left:18px}.ai li{margin:3px 0}.ai li::marker{color:var(--cy)}
  .ai strong{color:#fff;font-weight:600}
  .ai a{color:var(--cy);text-underline-offset:3px}
  .ai code{font-size:.9em;padding:1px 5px;border-radius:5px;background:var(--bg2);border:1px solid var(--line)}
  .src{margin-top:8px;font-size:12px;color:var(--mu);line-height:1.45}
  .src span{color:#C9C9EE}
  .err{border-left-color:var(--or)}
  .caret{display:inline-block;width:7px;height:15px;background:var(--cy);vertical-align:-2px;margin-left:2px;animation:blink 1s steps(2) infinite}
  @keyframes blink{50%{opacity:0}}
  .note{font-size:12.5px;color:var(--mu);padding-left:16px}

  .chips{display:flex;flex-wrap:wrap;gap:7px;padding:2px 0 4px}
  .chip{font-size:13px;padding:7px 12px;border-radius:999px;border:1px solid var(--line2);background:rgba(17,10,58,.7);
    color:var(--tx);text-align:left;transition:border-color .2s,color .2s,background .2s}
  .chip:hover{border-color:var(--cy);color:var(--cy);background:rgba(0,245,200,.06)}

  form{display:flex;align-items:flex-end;gap:8px;margin:8px 12px 10px;padding:6px 6px 6px 14px;border:1px solid var(--line2);
    border-radius:16px;background:var(--bg2);transition:border-color .2s}
  form:focus-within{border-color:var(--cy)}
  textarea{flex:1;resize:none;border:0;outline:0;background:none;color:var(--tx);font:inherit;font-size:14.5px;line-height:1.45;
    max-height:120px;padding:7px 0}
  textarea:focus-visible{outline:0}
  textarea::placeholder{color:#6E6E9C}
  .send{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;flex:none;color:#04010F;
    background:linear-gradient(135deg,var(--cy),#3B82F6);transition:opacity .2s,transform .2s}
  .send:disabled{opacity:.35;cursor:default}
  .send:not(:disabled):hover{transform:translateY(-1px)}
  .foot{padding:0 16px 12px;font-size:11.5px;color:#6E6E9C;text-align:center}
  .foot a{color:var(--mu)}

  @media (max-width:520px){
    :host{right:12px;bottom:12px}
    .panel{position:fixed;inset:0;width:auto;height:auto;border-radius:0;border:0}
    .launcher small{display:none}
  }
  @media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
  `;

  // ---------- tiny safe markdown ----------
  function esc(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function inline(s) {
    s = esc(s);
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?])/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
    s = s.replace(/(^|[\s(])([\w.+-]+@[\w-]+\.[\w.]+[\w])/g, '$1<a href="mailto:$2">$2</a>');
    return s;
  }
  function md(text) {
    var out = [], list = null;
    text.split(/\n/).forEach(function (line) {
      var m = line.match(/^\s*(?:[-*•]|\d+[.)])\s+(.*)$/);
      if (m) { if (!list) { list = []; } list.push("<li>" + inline(m[1]) + "</li>"); return; }
      if (list) { out.push("<ul>" + list.join("") + "</ul>"); list = null; }
      if (line.trim()) out.push("<p>" + inline(line.replace(/^#+\s*/, "")) + "</p>");
    });
    if (list) out.push("<ul>" + list.join("") + "</ul>");
    return out.join("");
  }

  // ---------- DOM ----------
  var host = document.createElement("div");
  host.id = "ask-saurav";
  var root = host.attachShadow({ mode: "open" });
  root.innerHTML =
    "<style>" + CSS + "</style>" +
    '<div class="teaser" hidden role="status"><button class="x" aria-label="Dismiss">×</button>Recruiter or professor? Ask my AI about my research, projects or experience.</div>' +
    '<button class="launcher" aria-label="Open chat with Saurav\'s AI assistant" aria-expanded="false">' +
      '<span class="ring"><span>SK</span></span><span><b>Ask Saurav</b><small>AI trained on my work</small></span></button>' +
    '<section class="panel" hidden role="dialog" aria-label="Chat with Saurav\'s AI assistant">' +
      '<header><span class="ring"><span>SK</span></span><div class="who"><h2>Saurav Kumar</h2>' +
        '<p><span class="dot"></span><span class="status">AI version of me</span></p></div>' +
        '<button class="iconbtn reset" aria-label="Start a new chat" title="New chat"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/></svg></button>' +
        '<button class="iconbtn close" aria-label="Close chat" title="Close"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg></button>' +
      "</header>" +
      '<div class="log" aria-live="polite"></div>' +
      '<form><label for="q" style="position:absolute;left:-9999px">Your question</label>' +
        '<textarea id="q" rows="1" maxlength="500" placeholder="Ask about my projects, research, skills…"></textarea>' +
        '<button class="send" type="submit" aria-label="Send" disabled><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button></form>' +
      '<p class="foot">AI can make mistakes. For anything important, email <a href="mailto:sauravsuz@gmail.com">sauravsuz@gmail.com</a></p>' +
    "</section>";

  var $ = function (s) { return root.querySelector(s); };
  var launcher = $(".launcher"), panel = $(".panel"), log = $(".log"), form = $("form"),
      input = $("textarea"), send = $(".send"), teaser = $(".teaser"), statusEl = $(".status"), dot = $(".dot");

  function mount() { document.body.appendChild(host); }
  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);

  // ---------- server warm-up (free Hugging Face Spaces sleep when idle) ----------
  var warming = null;
  function warmUp() {
    if (serverReady) return Promise.resolve();
    if (warming) return warming;
    var slow = setTimeout(function () { setStatus("Waking up the server. This can take up to a minute the first time.", true); }, 2500);
    warming = (function attempt(n) {
      return fetch(API + "/health", { cache: "no-store" })
        .then(function (r) { if (!r.ok) throw new Error(r.status); })
        .then(function () { serverReady = true; clearTimeout(slow); setStatus(); })
        .catch(function () {
          if (n <= 0) { clearTimeout(slow); warming = null; setStatus("Server is offline right now. Email me at sauravsuz@gmail.com.", true); throw new Error("offline"); }
          return new Promise(function (res) { setTimeout(res, 4000); }).then(function () { return attempt(n - 1); });
        });
    })(20);
    return warming;
  }
  function setStatus(text, warn) {
    statusEl.textContent = text || "AI version of me";
    dot.classList.toggle("wait", !!warn);
  }

  // ---------- rendering ----------
  function scroll() { log.scrollTop = log.scrollHeight; }
  function add(cls, html) {
    var el = document.createElement("div");
    el.className = "msg " + cls;
    el.innerHTML = html;
    log.appendChild(el); scroll();
    return el;
  }
  function renderSources(el, sources) {
    if (!sources || !sources.length) return;
    var names = sources.slice(0, 3).map(function (s) { return "<span>" + esc(s.section.split(" › ").pop().split(" — ")[0]) + "</span>"; });
    var p = document.createElement("p");
    p.className = "src";
    p.innerHTML = "From: " + names.join(", ");
    el.appendChild(p);
  }
  function showChips(list) {
    var wrap = document.createElement("div");
    wrap.className = "chips";
    list.forEach(function (q) {
      var b = document.createElement("button");
      b.className = "chip"; b.type = "button"; b.textContent = q;
      b.onclick = function () { wrap.remove(); ask(q); };
      wrap.appendChild(b);
    });
    log.appendChild(wrap); scroll();
  }
  function intro() {
    log.innerHTML = "";
    add("ai", md(GREETING));
    var fallback = FALLBACK_SUGGESTIONS;
    warmUp()
      .then(function () { return fetch(API + "/suggestions").then(function (r) { return r.json(); }); })
      .then(function (d) { showChips((d && d.suggestions || fallback).slice(0, 6)); })
      .catch(function () { showChips(fallback); });
  }

  // ---------- chat (Server-Sent Events over fetch) ----------
  function ask(question) {
    question = (question || "").trim();
    if (!question || busy) return;
    busy = true; send.disabled = true; host.classList.add("thinking");
    root.querySelectorAll(".chips").forEach(function (c) { c.remove(); });
    add("me", esc(question));
    var bubble = add("ai", '<span class="caret"></span>');
    var text = "", sources = [], failed = false;

    warmUp()
      .then(function () {
        return fetch(API + "/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question, history: history.slice(-12), session_id: sessionId }),
        });
      })
      .then(function (res) {
        if (res.status === 429) throw new Error("You're asking faster than I can think. Wait a minute and try again.");
        if (res.status === 422) throw new Error("Please keep your question under 500 characters.");
        if (!res.ok || !res.body) throw new Error("I couldn't reach my server. Try again, or email me at sauravsuz@gmail.com.");
        var reader = res.body.getReader(), decoder = new TextDecoder(), buf = "";
        function pump() {
          return reader.read().then(function (r) {
            if (r.done) return;
            buf += decoder.decode(r.value, { stream: true });
            var parts = buf.split("\n\n"); buf = parts.pop();
            parts.forEach(function (chunk) {
              chunk.split("\n").forEach(function (line) {
                if (line.indexOf("data: ") !== 0) return;
                var ev; try { ev = JSON.parse(line.slice(6)); } catch (e) { return; }
                if (ev.type === "sources") sources = ev.sources;
                else if (ev.type === "token") { text += ev.text; bubble.innerHTML = md(text) + '<span class="caret"></span>'; scroll(); }
                else if (ev.type === "error") { failed = true; text = ev.text; }
              });
            });
            return pump();
          });
        }
        return pump();
      })
      .then(function () {
        bubble.innerHTML = md(text || "I don't have an answer for that. Try rephrasing, or email me at sauravsuz@gmail.com.");
        if (failed) bubble.classList.add("err");
        else {
          renderSources(bubble, sources);
          history.push({ role: "user", content: question }, { role: "assistant", content: text });
        }
      })
      .catch(function (err) {
        bubble.classList.add("err");
        bubble.innerHTML = md(err && err.message && err.message !== "offline" ? err.message
          : "My server is offline right now. Email me at sauravsuz@gmail.com and I'll reply personally.");
      })
      .then(function () {
        busy = false; host.classList.remove("thinking");
        send.disabled = !input.value.trim(); scroll(); input.focus();
      });
  }

  // ---------- open / close ----------
  var started = false;
  function open(question) {
    teaser.hidden = true;
    panel.hidden = false; launcher.hidden = true; launcher.setAttribute("aria-expanded", "true");
    if (!started) { started = true; intro(); }
    if (question) ask(question);
    setTimeout(function () { input.focus(); }, 50);
  }
  function close() {
    panel.hidden = true; launcher.hidden = false; launcher.setAttribute("aria-expanded", "false"); launcher.focus();
  }

  launcher.onclick = function () { open(); };
  $(".close").onclick = close;
  $(".reset").onclick = function () { if (busy) return; history = []; intro(); input.focus(); };
  teaser.querySelector(".x").onclick = function (e) { e.stopPropagation(); teaser.hidden = true; try { sessionStorage.setItem("ask-saurav-teased", "1"); } catch (_) {} };
  teaser.onclick = function () { open(); };
  panel.addEventListener("keydown", function (e) { if (e.key === "Escape") close(); });
  input.addEventListener("input", function () {
    send.disabled = busy || !input.value.trim();
    input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 120) + "px";
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit ? form.requestSubmit() : form.onsubmit(e); }
  });
  form.onsubmit = function (e) {
    e.preventDefault();
    var q = input.value; input.value = ""; input.style.height = "auto"; send.disabled = true; ask(q);
  };
  document.addEventListener("click", function (e) {
    var t = e.target && e.target.closest && e.target.closest("[data-ask-saurav]");
    if (t) { e.preventDefault(); open(t.getAttribute("data-ask-saurav") || ""); }
  });

  // Gentle hint once per visit; also starts waking the free server early.
  setTimeout(function () { warmUp().catch(function () {}); }, 1500);
  if (ds.teaser !== "false") {
    setTimeout(function () {
      var seen = false; try { seen = sessionStorage.getItem("ask-saurav-teased"); } catch (_) {}
      if (!seen && panel.hidden) teaser.hidden = false;
    }, 6000);
  }

  window.AskSaurav = { open: open, close: close, ask: function (q) { open(q); } };
})();
