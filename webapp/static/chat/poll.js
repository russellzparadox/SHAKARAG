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
