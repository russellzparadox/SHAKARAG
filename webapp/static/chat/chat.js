(function () {
  const list = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  if (!form || !list) return;

  const input = document.getElementById("question-input");
  const btn = document.getElementById("send-btn");
  const csrf = form.querySelector("[name=csrfmiddlewaretoken]").value;

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function bubble(cls, html) {
    const div = document.createElement("div");
    div.className = "bubble " + cls;
    div.innerHTML = html;
    list.appendChild(div);
    list.scrollTop = list.scrollHeight;
    return div;
  }

  function resultBlock(meta) {
    let html = '<div class="meta-block">';
    if (meta.sql) {
      html += `<details><summary>SQL</summary><pre>${esc(meta.sql)}</pre></details>`;
    }
    if (meta.columns && meta.columns.length) {
      const cap = Math.min(meta.rows.length, 25);
      html += `<details open><summary>Results (${meta.row_count} rows)</summary><div class="table-scroll"><table><thead><tr>`;
      for (const c of meta.columns) html += `<th>${esc(c)}</th>`;
      html += "</tr></thead><tbody>";
      for (const row of meta.rows.slice(0, cap)) {
        html += "<tr>";
        for (const v of row) html += `<td>${v === null ? "NULL" : esc(String(v).slice(0, 80))}</td>`;
        html += "</tr>";
      }
      html += "</tbody></table></div>";
      if (meta.truncated) html += "<small>results capped</small>";
      html += "</details>";
    }
    if (meta.tables_used && meta.tables_used.length) {
      html += '<small class="chips">' + meta.tables_used.map((t) => `<span class="chip">${esc(t)}</span>`).join("") + "</small>";
    }
    return html + "</div>";
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    bubble("user", esc(question));
    input.value = "";
    btn.disabled = true;
    const pending = bubble("assistant", "<em>thinking…</em>");

    try {
      const resp = await fetch(list.dataset.sendUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrf },
        body: JSON.stringify({ question }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || resp.statusText);
      pending.innerHTML =
        esc(data.assistant.content).replace(/\n/g, "<br>") +
        resultBlock(data.assistant.meta || {}) +
        feedbackBlock(data.assistant.id);
      wireFeedback(pending);
    } catch (err) {
      pending.innerHTML = `⚠️ ${esc(err.message)}`;
    } finally {
      btn.disabled = false;
      input.focus();
      list.scrollTop = list.scrollHeight;
    }
  });

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
})();
