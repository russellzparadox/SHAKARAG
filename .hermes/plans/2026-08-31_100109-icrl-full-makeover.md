# ICRL Full Makeover — Make SHAKARAG Match the Paper

> **For Hermes:** Use subagent-driven-development to execute task-by-task after plan approval.

**Goal:** Bring SHAKARAG's ICRL/RAG-over-SQL pipeline to faithful, production-quality
parity with *In-Context Reinforcement Learning based RAG for Text-to-SQL* (Toteja et al.),
and measurably improve schema-recall + SQL-execution accuracy on the live ShakaERP warehouse.

**Architecture:** Keep the layered architecture (`dialects → chunking → store →
icrl → pipeline → webapp`); surgically upgrade each layer toward paper-faithful
semantics, then re-wire pipeline to use the upgraded ICRL KB at both schema-pooling
and few-shot time. Backward-compatible: existing collections, examples, and learned
few-shots continue to work; behaviour change is opt-in per profile.

**Tech Stack:** Python 3.11, ChromaDB, OpenAI-compatible LLM, Django 5 (webapp),
pytest.

---

## 0. Current state vs paper (gap analysis)

| Paper component | Current code | Gap |
|---|---|---|
| **Graph construction** (§A.2, Alg. A.1) | `build_schema_graph` exists but uses bare names + treats FK as undirected only | Root `R`→DB→Table→FK edges collapsed; DB-level structure ignored; no FK-directionality; no many-to-many detection |
| **Serialisation** (Alg. A.2) | `random_walk_traversals` — pure random walks, no cutoff depth, no power-set branching, no revisit handling | Algorithm A.2 calls for depth-cutoff `k`, power-set branching when type-3 (FK) edge has >1 children, visited-set per path. None of this exists. |
| **Traversal diversity** | Random sampling biased to fact starts only | Misses dimension-only traversals, dimension→dimension (snowflake) walks, bridge tables |
| **Complexity reward** (§A.3) | Implemented; uses B1..B4 keyword buckets | Match, but per-iteration it gives no feedback on *which bucket is weak* — coach prompt is generic |
| **Coach feedback LLM** (§3.1) | Generic "make it harder" prompt | Paper: feedback receives reward signal + bucket counts. We pass counts, but coach has no per-bucket target list of "easy→hard operators" to suggest. |
| **ICRL termination** | Fixed max-iterations OR `min_reward` | Paper: iterate until reward plateau (Δ < ε) OR max iters. Plateau detection missing → wasted LLM calls or premature stop. |
| **SQL validation in loop** | `validate_sql` rejects DML but doesn't *execute* | Paper validates executability — we should at minimum AST-parse + run against a throwaway DDL fixture, or do a static read-only EXPLAIN. |
| **Router KB** | `<collection>-router` exists, distances used | OK structurally; needs: weighted vote (✓), normalisation across collections, distance-aware gating to avoid false positives from poor embeddings |
| **LLM-aided schema pooling** (§4) | `_llm_schema_pool` exists; top-24 → LLM picks | Paper: top-k then LLM picks from *k*, not from `3·k`; also picks should retain candidates when LLM is unsure (✓ we fall back); we should not prune below `max(2, ranked/3)` — already do |
| **Few-shot (q, D, S) examples** | `retrieve_synthetic_example` returns single nearest with `max_distance=0.9` | Paper uses k=1–5; distance gate (0.9) is brittle — needs cosine-aware gating, table-overlap-aware fallback to verified examples |
| **Validation/eval** | None | No recall/EX metrics; no held-out Spider/Bird-style benchmark for the live warehouse |
| **Persistence** | `webapp/icrl/db<pk>.json` written | OK; missing: incremental re-index without re-running LLM, dedupe by `(question_hash, tables_signature)` |
| **Webapp wiring** | `scripts/run_icrl.py` only | No UI hook to view router KB, trigger rebuild, see coverage stats |

---

## 1. Proposed approach

### 1.1 Faithful Alg. A.1 + A.2 in `rag/icrl.py`

Replace `build_schema_graph` + `random_walk_traversals` with:

- **`SchemaGraph`** class that holds three typed edge types:
  - type-1: `R` (root) → `DB` (database)
  - type-2: `DB` → `Table`
  - type-3: `Table` → `Table` (FK, **directional**, with reverse edges for navigation)
- **`enumerate_traversals(graph, cutoff=k)`** implementing paper Algorithm A.2:
  - depth-first from root
  - at each type-3 fan-out (>1 children), expand via `power_set` (skip empty + full)
  - visited-set per path
  - yield paths of length 1..k
- **Deterministic + bounded** — no random sampling, runs once per schema
- Filter out traversals shorter than 2 tables (paper Eq. 1+ requires joins)
- Cap total traversals at e.g. 10k; if more, take a stratified sample (1 per DB, 1 per table-as-start)

### 1.2 Faithful reward + coach (§3.1)

- Split `complexity_reward` into two public functions:
  - `complexity_reward(sql)` → `(score, bucket_counts)` (existing behaviour)
  - `bucket_gaps(counts)` → ordered list of weakest buckets (e.g. `["aggregation", "conditional"]`)
- Rewrite `FEEDBACK_SYSTEM` to include:
  - the bucket gaps (so coach knows what's missing)
  - a curated **operator-suggestion table** per bucket (paper A.3 — `JOIN`→`LEFT JOIN`+`IS NULL`, `GROUP BY`→`HAVING`, `WHERE`→`CASE WHEN`/`IN (...)`)
- Add **plateau termination** in `ICRLGenerator.run`:
  - track reward history, stop when `max(reward) - max(prev_best) < ε` for 1 iteration
  - expose `plateau_epsilon` config (default 0.5)
- Add **executability gate** before scoring:
  - static parse via `sqlglot` if available (paper doesn't mandate, but it eliminates malformed SQL that scores high by keyword density)
  - fall back to existing `validate_sql` when `sqlglot` is not installed (don't break current behaviour)

### 1.3 Router KB upgrade

- **`index_qa_triplets`**:
  - deterministic ID: `sha1(f"{schema}.{tables_sorted}::{question_normalized}")` (so re-runs dedupe)
  - metadata: add `dialect`, `traversal_signature`, `iteration_count`, `reward_buckets` (JSON), `executable=True`
- **`retrieve_tables_for_question`**:
  - gate by `distance < 0.85` AND `reward ≥ MIN_REWARD` (currently no reward gate)
  - weighted vote with `weight = (1 - distance) * reward` (currently `1 - distance` only)
  - normalise scores by sum so output is a *probability* over tables → enables threshold cut
- **`retrieve_synthetic_example`**:
  - return top-K (k=1–3) instead of top-1
  - add `table_overlap_with_question` heuristic: boost examples whose tables match the candidate tables from `_llm_schema_pool`

### 1.4 Pipeline rewiring

- `RagPipeline.retrieve` — add a single ICRL call **before** the embedder search:
  - quick LLM-aided table-router that uses **only** the router KB (no schema chunks yet)
  - its output becomes a **seed** for the embedder search query (`expand_question` now also includes router tables)
- `RagPipeline.generate_sql` — use ICRL examples in addition to verified examples:
  - inject up to 2 ICRL examples *and* up to 2 verified examples (instead of "ICRL only when verified is empty")
  - format: `ICRL examples` block + `VERIFIED examples` block, clearly labelled
- `_llm_schema_pool`:
  - reduce candidate window from 24 → paper-faithful `top_k` (default 8)
  - allow multi-round: pass current picked tables back to LLM for confirmation if `len(picks) < 2`
- Add **batch executor** for the ICRL loop: `ICRLGenerator.run_batch(traversals, concurrency=4)` — `concurrent.futures.ThreadPoolExecutor` because each traversal is a sequential LLM chain.

### 1.5 Persistence & dedupe

- `run_icrl_generation` becomes **idempotent**:
  - on start, read `webapp/icrl/db<pk>.json` (if present) and the router KB
  - skip traversals whose deterministic signature already has an entry with `reward ≥ MIN_REWARD`
  - on finish, merge + write back
- Add `--incremental` flag (default ON) to `scripts/run_icrl.py`
- Add `--dry-run` flag that prints traversal count + estimated LLM cost (no calls)

### 1.6 Evaluation harness (paper §4 metrics)

New `tests/test_icrl_eval.py` and `scripts/eval_recall.py`:

- **Schema recall (R@k)** — for each gold question in a held-out file:
  - retrieve from router KB
  - compare top-k tables vs gold table set
  - report R@1, R@2, R@5, R@10
- **Execution accuracy (EX)** — generate SQL for held-out questions, execute, compare result rows/columns with gold
- Generate the held-out set on demand by:
  1. introspect warehouse
  2. for a sample of tables, run deterministic NL templates (e.g. "count rows", "sum amount per category for last 30 days")
  3. capture (question, SQL, expected_row_count) tuples
- Report metrics to `webapp/icrl/eval/db<pk>.json` + console

### 1.7 Webapp wiring

- `webapp/chat/views.py` + `rag_service.py`:
  - new endpoint `POST /chat/<dbpk>/icrl/rebuild/` — kicks `_reindex_icrl_worker` thread (mirrors `_reindex_worker`)
  - new endpoint `GET /chat/<dbpk>/icrl/status/` — returns `{count, last_run, coverage_pct, avg_reward}`
  - new endpoint `GET /chat/<dbpk>/icrl/sample/?k=5` — returns random router entries for inspection
  - template addition: a small "Synthetic training data" panel in the chat sidebar showing the above
- Add `ICRLCoverage` denormalised field on `DatabaseProfile`:
  - `icrl_count`, `icrl_avg_reward`, `icrl_last_run`, `icrl_coverage_pct = icrl_count / table_count * 100`
  - updated by the worker

### 1.8 Verification strategy

- Unit tests: graph + traversals (deterministic on fixture schemas), reward (parity with paper examples A.3), plateau logic, dedupe IDs, pool window sizing
- Integration test: with `FakeLLM`, full `ICRLGenerator.run` for 3-table fixture produces non-trivial SQL + correct termination
- Eval harness: run on the live ShakaERP warehouse (per profile), target R@1 improvement vs baseline (current `scripts/run_icrl.py` output)
- Backward-compat: existing collections, examples, learned few-shots remain readable; new ICRL code reads both old and new `db<pk>.json` shapes (with a `version` key added)

---

## 2. Step-by-step plan

> Tasks are bite-sized (2–5 min each). Every code task = TDD: failing test → implementation → passing test → commit. Use subagent-driven-development.

### Phase A — Faithful graph & traversal (paper §A.1, A.2)

**Task A1.** Add `tests/test_icrl_graph.py` skeleton + create `rag/icrl_graph.py` module file (empty, with module docstring quoting Alg. A.1+A.2). Run `pytest tests/test_icrl_graph.py -q` — expect collection-only PASS.

**Task A2.** Add test: `test_algorithm_a1_root_db_table_fk_edges`. Build a 2-database, 4-table fixture with 1 FK. Assert graph has `R→DB`, `DB→Table`, `Table→Table(FK)` edges typed 1/2/3. Run → FAIL (no implementation).

**Task A3.** Implement `SchemaGraph` dataclass with `nodes`, `edges` (list of `(src, dst, type)` tuples) and `add_node`/`add_edge`. Minimal — only what test A2 needs. Run → PASS.

**Task A4.** Add test: `test_algorithm_a1_fk_directional_with_reverse`. Same fixture; assert FK edge has `(src→dst, type=3)` AND a virtual reverse for navigation but stored directionally. Run → FAIL.

**Task A5.** Implement `add_fk_pair` that adds both directional edge and a reverse navigation edge marked `is_reverse=True`. Run → PASS.

**Task A6.** Add test: `test_algorithm_a2_traversals_cutoff_depth`. With a 4-node graph (A-B-C-D chain via type-3), cutoff=2 → expect traversals `[R→DB→A]`, `[R→DB→B]`, `[R→DB→C]`, `[R→DB→D]`, `[R→DB→A→B]`, `[R→DB→B→C]`, `[R→DB→C→D]`. Run → FAIL.

**Task A7.** Implement `enumerate_traversals(graph, cutoff)` per paper Alg. A.2 verbatim (depth-first, type-3 power-set branching, visited per path). Run → PASS.

**Task A8.** Add test: `test_algorithm_a2_power_set_branching`. Hub table T with 3 dim children — at depth ≥1, expect power-set traversal of (1, 2) and (2, 3) subsets. Run → FAIL.

**Task A9.** Wire power-set expansion in `enumerate_traversals`. Run → PASS.

**Task A10.** Add test: `test_algorithm_a2_no_cycles_within_path`. Cycle A→B→A via type-3 reverse; expect no traversal `[A, B, A]`. Run → FAIL (current impl doesn't track visited).

**Task A11.** Add per-path `visited` set in DFS recursion. Run → PASS.

**Task A12.** Add test: `test_enumerate_traversals_deterministic_and_bounded`. Same input → same output, no randomness. Run → PASS (already deterministic).

**Task A13.** Add `rag/icrl_graph.py` → re-export `SchemaGraph`, `enumerate_traversals` from `rag/icrl.py` for backwards compatibility (keep old names as deprecated wrappers). Run all existing `tests/test_icrl.py` — must still pass.

### Phase B — Faithful reward + coach (§3.1, §A.3)

**Task B1.** Add `tests/test_icrl_reward.py`. Test: `test_reward_bucket_normalisation_parity_with_paper_a3`. Compute reward for SQL `SELECT d.title, SUM(f.amount) ... GROUP BY d.title HAVING SUM(f.amount) > 100 ORDER BY total DESC`; assert counts match expected per-bucket values from paper §A.3.

**Task B2.** Verify current `complexity_reward` produces matching counts. If not, fix the keyword regex (e.g. `JOIN` matching `INNER JOIN`).

**Task B3.** Add test: `test_bucket_gaps_orders_by_weakest`. SQL `SELECT a FROM t` → gaps `["aggregation", "conditional", "retrieval", "modification"]`. Run → FAIL.

**Task B4.** Implement `bucket_gaps(counts) -> list[str]`. Run → PASS.

**Task B5.** Add test: `test_feedback_prompt_contains_bucket_gaps_and_suggestions`. Mock LLM, capture user message, assert it includes `aggregation`, `JOIN`, `GROUP BY`, `HAVING` and a `bucket_gaps` JSON block. Run → FAIL.

**Task B6.** Rewrite `FEEDBACK_SYSTEM` + `_feedback` to inject `bucket_gaps(counts)` + a static `OPERATOR_SUGGESTIONS` dict from paper §A.3. Run → PASS.

**Task B7.** Add test: `test_plateau_termination`. FakeLLM returns improving rewards `[1.0, 2.0, 2.1]`; expect stop at iter 3 (Δ < ε=0.5). Run → FAIL.

**Task B8.** Implement plateau detection in `ICRLGenerator.run`: track `best_so_far` per iter, stop when improvement < `plateau_epsilon`. Run → PASS.

**Task B9.** Add test: `test_max_iterations_caps_plateau_loop`. FakeLLM returns constant reward; expect `iterations ≤ max_iterations`. Run → PASS.

**Task B10.** Add test: `test_executable_sql_required_for_reward_score`. Malformed SQL like `SELECT a FROM JOIN b ON x =` raises sqlglot parse error → reward not computed, iteration retried with feedback. Run → FAIL.

**Task B11.** Implement `parse_sql(sql)` helper using `sqlglot` if installed (already in requirements — verify), else fall back to no-op. Wire into `ICRLGenerator.run` before `complexity_reward`. Add `sqlglot` to `requirements.txt` if missing.

### Phase C — Router KB upgrade

**Task C1.** Add test: `test_index_qa_deterministic_ids`. Same question+tables → same ID on re-index. Run → FAIL.

**Task C2.** Implement deterministic ID via `hashlib.sha1(f"{tables_sorted}::{q_normalized}").hexdigest()[:16]`. Run → PASS.

**Task C3.** Add test: `test_index_qa_metadata_includes_buckets_and_executable`. Inspect upserted metadata. Run → FAIL.

**Task C4.** Add `reward_buckets`, `executable`, `dialect` to metadata. Run → PASS.

**Task C5.** Add test: `test_retrieve_tables_gates_on_distance_and_reward`. Index 2 entries — one near, one far; assert far one is gated. Run → FAIL.

**Task C6.** Implement `distance < 0.85 AND reward ≥ MIN_REWARD` gate. Run → PASS.

**Task C7.** Add test: `test_retrieve_tables_weighted_by_distance_and_reward`. Manually verify `weight = (1-d) * reward`. Run → PASS after implementation tweak.

**Task C8.** Add test: `test_retrieve_synthetic_returns_top_k`. `k=3` returns 3 entries ordered by score. Run → FAIL.

**Task C9.** Implement top-K in `retrieve_synthetic_example`. Run → PASS.

**Task C10.** Add test: `test_retrieve_synthetic_boosts_table_overlap`. Two candidates with same distance; the one whose tables overlap with `_llm_schema_pool` candidate list wins. Run → FAIL.

**Task C11.** Add `boost_table_overlap` parameter to `retrieve_synthetic_example`; multiply weight by `1 + 0.2 * overlap_count`. Run → PASS.

### Phase D — Pipeline rewiring

**Task D1.** Add test: `test_pipeline_retrieve_seeds_query_with_router_tables`. Mock router KB → assert `expand_question` is called with router tables appended. Run → FAIL.

**Task D2.** Move ICRL router call to the top of `RagPipeline.retrieve`, pass output to `expand_question`. Run → PASS.

**Task D3.** Add test: `test_pipeline_generate_sql_injects_both_icrl_and_verified`. Mock both stores; assert prompt contains both `ICRL example` and `VERIFIED example` blocks. Run → FAIL.

**Task D4.** Modify `_few_shot_block` to merge ICRL + verified, label separately. Run → PASS.

**Task D5.** Add test: `test_llm_schema_pool_uses_top_k_window`. Top-8 candidates passed (not top-24). Run → FAIL.

**Task D6.** Change `_llm_schema_pool` slice from `[:24]` to `[:top_k or self.settings.top_k]`. Run → PASS.

**Task D7.** Add test: `test_llm_schema_pool_multi_round_when_unsure`. LLM returns empty `tables: []` first time → re-prompt with "if unsure, pick top 3 by table-name similarity". Run → FAIL.

**Task D8.** Implement multi-round schema pool: if picks empty, re-prompt with similarity hint. Run → PASS.

**Task D9.** Add test: `test_icrl_generator_batch_concurrent`. 4 traversals, FakeLLM, `concurrency=4` → total time ≤ 1× single (sanity). Run → FAIL.

**Task D10.** Implement `ICRLGenerator.run_batch` with `ThreadPoolExecutor`. Run → PASS.

### Phase E — Persistence & dedupe

**Task E1.** Add test: `test_run_icrl_idempotent_skips_existing`. Pre-populate `db<pk>.json` with 3 entries; run with `--n 10`; assert only 7 new entries. Run → FAIL.

**Task E2.** Implement skip-if-exists logic in `run_icrl_generation`. Run → PASS.

**Task E3.** Add `--incremental` flag to `scripts/run_icrl.py` (default True). Already structurally trivial; no test needed.

**Task E4.** Add test: `test_run_icrl_dry_run_no_llm_calls`. `--dry-run` with FakeLLM that records calls; assert zero calls. Run → FAIL.

**Task E5.** Implement dry-run path in `run_icrl_generation` that returns after counting traversals. Run → PASS.

**Task E6.** Add `version` key to `db<pk>.json` schema (`{"version": 2, "entries": [...]}`). Add a reader that supports both v1 and v2. Backward compatible.

### Phase F — Evaluation harness

**Task F1.** Add `scripts/eval_recall.py` skeleton with `--db-profile` arg + argparse. Run `--help` → works.

**Task F2.** Add test: `test_recall_at_k_simple`. Build held-out (Q, gold_tables) set of 5; mock router with 3 known + 2 unknown. Assert R@1, R@2 match expected fractions.

**Task F3.** Implement `compute_recall(retrieved, gold, k=1|2|5|10)` in `rag/icrl_eval.py`.

**Task F4.** Add test: `test_holdout_generation_deterministic`. Same warehouse state → same held-out set. Run → PASS once deterministic seeding is added.

**Task F5.** Implement `generate_holdout(tables, n=20, seed=42)` in `rag/icrl_eval.py` — uses NL templates, captures `(question, sql_template, expected_count)`.

**Task F6.** Add test: `test_execution_accuracy_exact_match`. Mock execution returns known rows; assert EX = 1.0 when SQL yields same rows.

**Task F7.** Implement `execution_accuracy(generated_sql, executed, gold_rows)`.

**Task F8.** Wire `scripts/eval_recall.py` → for profile: introspect → generate holdout → run ICRL → measure R@k + EX → write `webapp/icrl/eval/db<pk>.json` → return exit code 0 if metrics meet threshold (e.g. R@1 ≥ 0.5), else 1.

### Phase G — Webapp wiring

**Task G1.** Add migration: `icrl_count`, `icrl_avg_reward`, `icrl_last_run`, `icrl_coverage_pct` on `DatabaseProfile`. Run `makemigrations` + `migrate`.

**Task G2.** Add `ICRLRebuildView` (POST `/chat/<dbpk>/icrl/rebuild/`) and `ICRLStatusView` (GET `/chat/<dbpk>/icrl/status/`) in `webapp/chat/views.py`. Mirror access rules of `_reindex`.

**Task G3.** Add `_reindex_icrl_worker` in `rag_service.py` — runs `run_icrl_generation` in a thread, updates profile fields on completion.

**Task G4.** Add `ICRLSampleView` (GET `/chat/<dbpk>/icrl/sample/?k=5`) — returns random router entries from Chroma.

**Task G5.** Wire URLs in `webapp/chat/urls.py`.

**Task G6.** Add sidebar panel in `webapp/chat/templates/chat/sidebar.html` showing ICRL coverage + buttons. JS handler in `static/chat/chat.js` to call rebuild + poll status (reuse `poll.js` pattern).

**Task G7.** Add Django test: `test_icrl_rebuild_requires_edit_access`. Hit endpoint as non-editor → 403.

**Task G8.** Add Django test: `test_icrl_status_returns_metrics`. Mock worker → assert JSON shape.

### Phase H — End-to-end verification on live ShakaERP

**Task H1.** Run `pytest tests/ -q` — all green.

**Task H2.** Run `cd webapp && ../.venv/bin/python manage.py test chat` — all green.

**Task H3.** Run `scripts/eval_recall.py --db-profile <shaka>` — capture baseline R@k and EX (target: R@1 ≥ current value + 10%).

**Task H4.** Run `scripts/run_icrl.py --db-profile <shaka> --n 60 --max-iterations 4` — verify new traversal enumeration runs (not random walks). Compare reward distribution to old run.

**Task H5.** Smoke test `python scripts/ask.py "top 5 customers by total sales last quarter"` — verify pipeline still works, ICRL boost applied, fewer empty-result retries.

**Task H6.** Update `DEVELOPMENT.md`:
  - §3.4 Learning loop: document ICRL upgrade
  - §8 Extension recipes: replace existing ICRL recipe with new one (incremental, dry-run, eval)
  - §9 Limitations: move completed items out

**Task H7.** Commit per-phase with conventional messages: `feat(icrl): faithful Alg. A.1+A.2`, `feat(icrl): bucket-aware coach + plateau termination`, `feat(icrl): router dedupe + reward gate`, `feat(pipeline): ICRL seed in retrieve + merged few-shot`, `chore(icrl): idempotent run + dry-run`, `feat(eval): R@k + EX harness`, `feat(webapp): ICRL rebuild/status/sample UI`.

---

## 3. Files likely to change

**New:**
- `rag/icrl_graph.py` (Alg. A.1+A.2)
- `rag/icrl_eval.py` (R@k, EX, holdout generation)
- `scripts/eval_recall.py`
- `tests/test_icrl_graph.py`
- `tests/test_icrl_reward.py`
- `tests/test_icrl_router_v2.py`
- `tests/test_icrl_eval.py`
- `webapp/chat/migrations/00xx_icrl_fields.py`

**Modified:**
- `rag/icrl.py` (re-export, refactor `complexity_reward`, `FEEDBACK_SYSTEM`, `ICRLGenerator.run`, `index_qa_triplets`, `retrieve_tables_for_question`, `retrieve_synthetic_example`, `run_icrl_generation`)
- `rag/pipeline.py` (seed retrieve from router, merge few-shots, smaller pool window, multi-round pool)
- `scripts/run_icrl.py` (`--incremental`, `--dry-run`, `version`-aware JSON)
- `webapp/chat/models.py` (4 fields on `DatabaseProfile`)
- `webapp/chat/views.py` (3 new views)
- `webapp/chat/urls.py` (3 new routes)
- `webapp/chat/rag_service.py` (`_reindex_icrl_worker`)
- `webapp/chat/templates/chat/sidebar.html` (ICRL panel)
- `webapp/chat/static/chat/chat.js` (rebuild + sample handlers)
- `DEVELOPMENT.md`
- `requirements.txt` (add `sqlglot` if missing)

---

## 4. Tests / validation

- All existing `tests/test_icrl.py` must keep passing (Phase A wraps, Phases B–E extend).
- New `tests/test_icrl_graph.py` — graph + traversal correctness (8+ tests).
- New `tests/test_icrl_reward.py` — reward parity + bucket gaps + plateau (5+ tests).
- New `tests/test_icrl_router_v2.py` — dedupe IDs, reward gate, top-K, table-overlap boost (5+ tests).
- New `tests/test_icrl_eval.py` — R@k math, EX math, holdout determinism (4+ tests).
- Updated `tests/test_pipeline_learning.py` — new seed-in-retrieve + merged-few-shot behaviour (3 new tests).
- Webapp: `webapp/chat/tests.py` — 3 new tests (rebuild auth, status shape, sample endpoint).
- Live validation:
  - `scripts/eval_recall.py --db-profile <shaka>` returns R@1 ≥ 0.5, R@5 ≥ 0.8 (stretch: R@10 ≥ 0.95)
  - EX ≥ 0.6 on the generated holdout
  - `scripts/run_icrl.py --db-profile <shaka> --dry-run --n 100` finishes in <5s, prints deterministic traversal count
  - `scripts/ask.py` on 5 canned questions returns non-empty rows for ≥4

---

## 5. Risks, tradeoffs, open questions

- **Backward compatibility:** old `db<pk>.json` files have no `version` key. Reader must handle both shapes. Old router KB IDs used `re.sub(r"[^a-z0-9]+", "-", question.lower())[:80]` — collisions are possible on re-index with deterministic IDs. Resolution: keep old IDs read-only, only write new ones on `--drop`. *(Decision: implement dual reader; flag in commit msg.)*
- **Cost:** `enumerate_traversals` on a 1000-table graph with cutoff=4 can produce millions of traversals. Mitigation: cap at 10k with stratified sampling (documented in `enumerate_traversals` docstring). *(Open: agree on cutoff default — propose 3, configurable via `ICRL_CUTOFF` env var.)*
- **Plateau ε:** too tight (0.1) wastes LLM calls, too loose (2.0) stops too early. Default 0.5, expose via `ICRLGenerator(plateau_epsilon=...)`.
- **Concurrency:** `ThreadPoolExecutor` with `concurrency=4` on a local Ollama may oversubscribe. Make it configurable; default 2.
- **sqlglot dependency:** add to `requirements.txt`. If install fails in the docker container, fall back gracefully (skip executable gate, log warning).
- **Eval ground truth:** the live warehouse has no labeled Spider/Bird set. The templated holdout is a proxy. *(Open: should we ship a tiny human-curated Q-set for the ShakaERP warehouse? Recommend yes — Phase H8, but only after core is green.)*
- **No streaming:** unchanged from current roadmap §9.
- **Single SQLite tenant DB:** unchanged.

---

## 6. Out of scope (deferred)

- Spider/Bird benchmark ingestion (paper §4 datasets)
- Fine-tuning any LLM (paper explicitly avoids)
- Streaming answers
- Multi-tenant migration to Postgres
- MCP server / Playwright E2E

---

## 7. Execution handoff

When this plan is approved, execute via `subagent-driven-development`:

1. Dispatch one subagent per **Phase** (A through H), each with:
   - Full Phase task list (A1, A2, ...)
   - This plan's relevant section
   - The project root path + working directory
   - Instruction: TDD per task, commit per task, run `pytest tests/test_icrl*.py -q` after each phase
2. After each phase, parent reviews: spec compliance → code quality → proceed
3. Final phase (H) is run by parent itself (live warehouse needs user confirmation)

Estimated effort: 8 phases × ~12 tasks × 4 min = ~6 hours of agent work.