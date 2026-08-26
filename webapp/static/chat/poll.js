(function () {
  const tbody = document.querySelector("tbody[data-status-poll]");
  if (!tbody) return;

  async function poll() {
    let active = false;
    for (const tr of tbody.querySelectorAll("tr[data-pk]")) {
      const badge = tr.querySelector('[data-role="status"]');
      const vectors = tr.querySelector('[data-role="vectors"]');
      if (!badge) continue;
      try {
        const r = await fetch(`/db/${tr.dataset.pk}/status/`);
        if (!r.ok) continue;
        const d = await r.json();
        badge.textContent = { none: "Not indexed", indexing: "Indexing…", ready: "Ready", error: "Error" }[d.status] || d.status;
        badge.className = "badge st-" + d.status;
        vectors.textContent = d.vectors ? `${d.vectors} vecs` : "";
        if (d.status === "indexing") active = true;
      } catch (e) {}
    }
    if (active) setTimeout(poll, 2500);
  }

  poll();
})();

/* ---- document upload (PDF/Word/Excel) into a database collection ---- */
(function () {
  document.querySelectorAll(".doc-upload input[type=file]").forEach((inp) => {
    inp.addEventListener("change", async () => {
      const formEl = inp.closest(".doc-upload");
      const pk = formEl.dataset.pk;
      const file = inp.files[0];
      if (!file) return;
      const label = inp.closest(".doc-upload-label");
      const oldText = label.textContent;
      label.textContent = "⏳ …";
      label.style.pointerEvents = "none";

      const fd = new FormData();
      fd.append("file", file);
      try {
        const resp = await fetch(`/db/${pk}/docs/`, {
          method: "POST",
          headers: { "X-CSRFToken": formEl.querySelector("[name=csrfmiddlewaretoken]").value },
          body: fd,
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
          alert(data.error || resp.statusText);
          label.textContent = oldText;
        } else {
          label.textContent = `✓ ${data.chunks}`;
          setTimeout(() => { location.reload(); }, 1200);
        }
      } catch (e) {
        alert(e.message);
        label.textContent = oldText;
      } finally {
        label.style.pointerEvents = "";
        inp.value = "";
      }
    });
  });
})();
