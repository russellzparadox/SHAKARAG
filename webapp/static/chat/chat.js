(function () {
  const list = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  if (!form || !list) return;

  const main = document.getElementById("chat-main");
  const input = document.getElementById("question-input");
  const btn = document.getElementById("send-btn");
  const csrf = form.querySelector("[name=csrfmiddlewaretoken]").value;

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // Markdown rendering: marked + DOMPurify, loaded from local static files.
  if (window.marked) {
    marked.setOptions({ gfm: true, breaks: true });
  }
  function md(text) {
    const raw = String(text ?? "");
    if (!window.marked || !window.DOMPurify) return esc(raw).replace(/\n/g, "<br>");
    return DOMPurify.sanitize(marked.parse(raw), { ADD_ATTR: ["target"] });
  }

  function scrollDown() { list.scrollTop = list.scrollHeight; }

  function bubble(cls, html) {
    const div = document.createElement("div");
    div.className = "bubble " + cls;
    div.innerHTML = html;
    list.appendChild(div);
    scrollDown();
    return div;
  }

  function eyebrow(role, timeStr) {
    const label =
      role === "user" ? "<b>You</b>" :
      role === "assistant" ? "<b>ShakaRAG</b>" : "system";
    return `<div class="msg-eyebrow">${label}<span>${esc(timeStr || "")}</span></div>`;
  }

  function resultBlock(meta) {
    let html = '<div class="meta-block">';
    if (meta.sql) {
      html += `<details><summary>SQL</summary><pre>${esc(meta.sql)}</pre></details>`;
    }
    if (meta.columns && meta.columns.length) {
      html += `<details open><summary>Results (${meta.row_count} rows)</summary><div class="table-scroll"><table><thead><tr>`;
      for (const c of meta.columns) html += `<th>${esc(c)}</th>`;
      html += "</tr></thead><tbody>";
      for (const row of meta.rows) {
        html += "<tr>";
        for (const v of row) html += `<td>${v === null ? "NULL" : esc(String(v).slice(0, 80))}</td>`;
        html += "</tr>";
      }
      html += "</tbody></table></div>";
      if (meta.truncated) html += `<small>result capped at server limit — refine the query for more</small>`;
      html += "</details>";
    }
    if (meta.tables_used && meta.tables_used.length) {
      html += '<small class="chips">' + meta.tables_used.map((t) => `<span class="chip">${esc(t)}</span>`).join("") + "</small>";
    }
    return html + "</div>";
  }

  function clarifyBlock(meta) {
    let html = `<div class="clarify-live"><p class="clarify-q">🤔 ${esc(meta.clarify_question || "")}</p>`;
    if (meta.options && meta.options.length) {
      html += '<div class="chips">';
      for (const o of meta.options) {
        html += `<button class="chip chip-option" data-opt="${esc(o)}">${esc(o)}</button>`;
      }
      html += "</div>";
    }
    html += '<small class="muted">pick an option or type your own answer</small></div>';
    return html;
  }

  function wireClarify(root) {
    root.querySelectorAll(".chip-option").forEach((el) => {
      el.addEventListener("click", () => {
        input.value = el.dataset.opt;
        form.dispatchEvent(new Event("submit", { cancelable: true }));
      });
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question || btn.disabled) return;
    bubble("user", eyebrow("user", nowHM()) +
      '<div class="bubble-text md">' + md(question) + "</div>");
    input.value = "";
    btn.disabled = true;
    if (main) main.classList.add("thinking");
    const pending = bubble(
      "assistant",
      eyebrow("assistant") +
        '<div class="bubble-text"><span class="typing-dots"><i></i><i></i><i></i></span></div>'
    );
    scrollDown();

    try {
      const resp = await fetch(list.dataset.sendUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrf },
        body: JSON.stringify({ question }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.statusText);
      const a = data.assistant;
      if (a.meta && a.meta.type === "clarify") {
        pending.innerHTML = eyebrow("assistant") +
          '<div class="bubble-text md">' + md(a.content) + "</div>" +
          clarifyBlock(a.meta || {});
        wireClarify(pending);
      } else if (a.meta && a.meta.error) {
        pending.classList.add("error");
        pending.innerHTML = eyebrow("assistant") +
          '<div class="bubble-text">⚠️ ' + esc(a.content || a.meta.error) + "</div>";
      } else {
        const m = a.meta || {};
        let badge = "";
        if (m.route === "document") {
          badge = '<span class="route-badge doc">📄 ' +
            esc((m.doc_sources || []).join(", ") || "documents") + "</span>";
        }
        pending.innerHTML = eyebrow("assistant") +
          '<div class="bubble-text md">' + md(a.content) + "</div>" +
          badge +
          resultBlock(m) +
          feedbackBlock(a.id);
        wireFeedback(pending);
      }
    } catch (err) {
      pending.classList.add("error");
      pending.innerHTML = eyebrow("assistant") +
        '<div class="bubble-text">⚠️ ' + esc(err.message) + "</div>";
    } finally {
      btn.disabled = false;
      if (main) main.classList.remove("thinking");
      input.focus();
      scrollDown();
    }
  });

  function nowHM() {
    const d = new Date();
    return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
  }

  function feedbackBlock(mid) {
    return `<div class="fb" data-mid="${mid}">
      <button class="fb-btn" data-value="up" title="Good answer — remember this query">👍</button>
      <button class="fb-btn" data-value="down" title="Wrong — forget this query">👎</button>
    </div>`;
  }

  async function sendFeedback(container, value) {
    const mid = container.dataset.mid;
    const resp = await fetch(list.dataset.sendUrl.replace(/\/send\/$/, "/feedback/"), {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRFToken": csrf },
      body: JSON.stringify({ message_id: parseInt(mid, 10), value }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    container.innerHTML = data.value === "up"
      ? '<span class="fb-done">✓ saved as example for future questions</span>'
      : '<span class="fb-done">✗ example removed</span>';
  }

  function wireFeedback(root) {
    root.querySelectorAll(".fb .fb-btn").forEach((btnEl) => {
      btnEl.addEventListener("click", () => {
        sendFeedback(btnEl.closest(".fb"), btnEl.dataset.value).catch(
          (err) => alert(err.message)
        );
      });
    });
  }

  wireFeedback(document);

  // Render markdown in history messages (server outputs raw text into [data-raw-md],
  // Django auto-escapes it so .textContent is the original markdown source).
  document.querySelectorAll("[data-raw-md]").forEach((el) => {
    el.innerHTML = md(el.textContent);
  });
})();
