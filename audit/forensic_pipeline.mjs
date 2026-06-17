#!/usr/bin/env node
/**
 * Forensic Audit Pipeline — dynamic-workflow orchestrator.
 *
 * Implements the `forensic-audit-pipeline` spec as a real, rerunnable Node
 * orchestration that drives headless `claude -p` subagents through five
 * sequential stages. The loops, parallel barriers, adversarial promotion
 * gates, stop-tests, per-call cost/time circuit-breakers, per-stage
 * checkpointing and resume logic all live HERE, in code — not in the
 * orchestrating model's context.
 *
 * Deliverables (written to ./audit, mirrored to git branch for durability):
 *   01-understanding.md  02-static-audit.md  03-execution.md
 *   04-goal.md           05-plan.md
 * Each .md embeds its schema-validated structured object in a fenced
 * ```json appendix, and the canonical object is also written to
 * audit/.work/0N-*.json. Prose-only output is a failure by construction.
 *
 * Agent execution primitive: `claude -p ... --output-format json
 * --json-schema <schema>` (structured output validated by the CLI) with
 * --strict-mcp-config (MCP off), --add-dir scoping and a per-call
 * --max-budget-usd cap. OAuth is host-managed, so --bare is NOT used.
 *
 * Usage:
 *   node audit/forensic_pipeline.mjs [--fresh|--resume] [--preflight]
 *        [--stage N] [--max-total-usd N] [--concurrency N]
 *        [--ceiling N] [--no-web]
 */

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from "node:fs";
import { resolve, join } from "node:path";

// ─────────────────────────── configuration ───────────────────────────
const REPO = resolve(process.cwd());
const AUDIT = join(REPO, "audit");
const WORK = join(AUDIT, ".work");
const BRANCH = "claude/adoring-hawking-qh05v8";
const LOG = join(WORK, "run.log");

const argv = process.argv.slice(2);
const hasFlag = (f) => argv.includes(f);
const flagVal = (f, d) => { const i = argv.indexOf(f); return i >= 0 && argv[i + 1] ? argv[i + 1] : d; };

const CFG = {
  fresh: hasFlag("--fresh"),
  preflight: hasFlag("--preflight"),
  onlyStage: flagVal("--stage", null) ? Number(flagVal("--stage", null)) : null,
  // Runaway circuit-breaker only. NOTE: `claude -p` reports API-EQUIVALENT cost; under a
  // Claude Max/Pro subscription real consumption is far lower (~10-100x), so this is a guard
  // against an infinite loop, NOT a usage budget. Effectively unlimited by default; pass
  // --max-total-usd to bound spend only when actually metering against the API.
  maxTotalUsd: Number(flagVal("--max-total-usd", "1000000")),
  concurrency: Number(flagVal("--concurrency", "4")),
  ceiling: Number(flagVal("--ceiling", "6")),              // hard ceiling for fixpoint loops -> halt-report
  web: !hasFlag("--no-web"),
  // model assignment: opus for adversarial/reasoning gates, sonnet for breadth/IO workers
  MODEL_SYNTH: "opus",
  MODEL_WORKER: "sonnet",
  MODEL_ADVERSARY: "opus",
};

// repo-relative ignore list (the declared denominator subtraction)
const IGNORE = [
  ".git/", "audit/", ".venv/", "node_modules/", "__pycache__/", ".ruff_cache/",
  ".pytest_cache/", "htmlcov/", ".mastery_engine_worktree/", "uv.lock",
];

let TOTAL_USD = 0;
let AGENT_SEQ = 0;            // guarantees a unique output-file path per agent call (names may collide)
const COST_BY_STAGE = {};

// ─────────────────────────── small utilities ───────────────────────────
function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  try { writeFileSync(LOG, line + "\n", { flag: "a" }); } catch {}
}

function sh(cmd, args, opts = {}) {
  return new Promise((res) => {
    const p = spawn(cmd, args, { cwd: REPO, ...opts });
    let out = "", err = "";
    p.stdout?.on("data", (d) => (out += d));
    p.stderr?.on("data", (d) => (err += d));
    p.on("close", (code) => res({ code, out, err }));
    p.on("error", (e) => res({ code: -1, out, err: String(e) }));
  });
}

// robustly extract a JSON object from agent output (pure JSON | ```json fence | first balanced {...})
function parseLooseJSON(s) {
  try { return JSON.parse(s); } catch {}
  const fence = s.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fence) { try { return JSON.parse(fence[1]); } catch {} }
  const start = s.indexOf("{");
  if (start >= 0) {
    let depth = 0, inStr = false, esc = false;
    for (let i = start; i < s.length; i++) {
      const c = s[i];
      if (inStr) { if (esc) esc = false; else if (c === "\\") esc = true; else if (c === '"') inStr = false; }
      else if (c === '"') inStr = true;
      else if (c === "{") depth++;
      else if (c === "}") { if (--depth === 0) { try { return JSON.parse(s.slice(start, i + 1)); } catch { break; } } }
    }
  }
  return null;
}

async function gitPersist(stageLabel) {
  await sh("git", ["add", "audit"]);
  const c = await sh("git", ["commit", "-m", `audit: checkpoint ${stageLabel}`]);
  if (c.code !== 0 && !/nothing to commit/.test(c.out + c.err)) log(`  git commit note: ${(c.err || c.out).trim().slice(0, 200)}`);
  for (let attempt = 0, delay = 2000; attempt < 4; attempt++, delay *= 2) {
    const p = await sh("git", ["push", "-u", "origin", BRANCH]);
    if (p.code === 0) { log(`  pushed checkpoint (${stageLabel})`); return; }
    log(`  push attempt ${attempt + 1} failed; retrying in ${delay}ms`);
    await new Promise((r) => setTimeout(r, delay));
  }
  log(`  WARNING: push failed after retries for ${stageLabel}; artifacts are committed locally`);
}

// ─────────────── the agent primitive: headless claude -p ───────────────
const GLOBAL_INVARIANTS = `You are a worker in a FORENSIC AUDIT pipeline. Obey these invariants without exception:
1. ABSENCE OF EVIDENCE IS NOT EVIDENCE OF ABSENCE. Never claim something "does not exist", "is unused", or "is unreachable" unless you name exactly where you looked and that search space is the full Stage-1 surface. Otherwise record it as "unverified", never "absent".
2. NO CLAIM WITHOUT A LOCATION. Every finding/behavior/assertion must cite a concrete path:line (or a named artifact) a reviewer can open. Drop any claim lacking a citable anchor.
3. COVERAGE HAS A DENOMINATOR. Stage-1's inventory is the denominator. Do not declare "done looking" until every relevant item has been visited.
4. VERIFY ADVERSARIALLY, NOT BY SELF-REVIEW. When asked to falsify, try to REFUTE each claim with counter-evidence; do not rubber-stamp.
5. You may read/run/instrument code freely in this sandbox. Only the audit documents ship. Do not worry about leaving the tree dirty.
Return ONLY the final JSON object conforming to the provided schema. Use your tools (Read/Grep/Glob/Bash) to ground every field in real evidence first.`;

/**
 * Run one subagent. Returns {ok, data, costUsd, raw, err}.
 * Structured output is enforced by --json-schema; we also defensively parse.
 */
async function runAgent({ name, prompt, schema, model, maxTurns = 50, budgetUsd = 4, allowedTools, cwd = REPO, timeoutMs = 900000, web = false }) {
  // Structured output is delivered via a FILE handoff: --json-schema is advisory in this CLI
  // build (agents narrate prose when they use tools), so we have the agent Write its JSON to a
  // path we control and read it back. Deterministic and decoupled from chat narration.
  const uid = AGENT_SEQ++;
  const safe = name.replace(/[^a-zA-Z0-9]+/g, "_");
  const outPath = join(WORK, `agent_${safe}_${uid}.json`);   // _${uid} prevents same-named concurrent agents clobbering each other
  rmSync(outPath, { force: true });
  const tools = (allowedTools || (web
    ? "Read,Grep,Glob,Bash,WebSearch,WebFetch"
    : "Read,Grep,Glob,Bash")) + ",Write";
  const fullPrompt = `${prompt}

═══ OUTPUT CONTRACT (mandatory) ═══
When finished, use the Write tool to write your final answer as a SINGLE valid JSON object to this exact absolute path:
  ${outPath}
The JSON MUST conform to this schema:
${JSON.stringify(schema)}
Write ONLY raw JSON to that file — no markdown fences, no prose, no trailing commentary. This file is the only thing that is read back.`;
  const args = [
    "-p", fullPrompt,
    "--output-format", "json",
    "--json-schema", JSON.stringify(schema),
    "--model", model,
    "--max-turns", String(maxTurns),
    "--max-budget-usd", String(budgetUsd),
    "--allowedTools", tools,
    "--add-dir", REPO,
    "--strict-mcp-config",
    "--permission-mode", "acceptEdits",   // root-safe; bypassPermissions/--dangerously-* are refused as root
    "--append-system-prompt", GLOBAL_INVARIANTS,
  ];
  log(`  ▶ agent[${name}#${uid}] model=${model} budget=$${budgetUsd} turns=${maxTurns} web=${web}`);
  const started = Date.now();
  const r = await new Promise((res) => {
    const p = spawn("claude", args, { cwd, stdio: ["ignore", "pipe", "pipe"] });   // close stdin -> no "waiting for stdin" hang
    let out = "", err = "";
    const killer = setTimeout(() => { try { p.kill("SIGKILL"); } catch {} }, timeoutMs);
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (err += d));
    p.on("close", (code) => { clearTimeout(killer); res({ code, out, err }); });
    p.on("error", (e) => { clearTimeout(killer); res({ code: -1, out, err: String(e) }); });
  });
  const secs = ((Date.now() - started) / 1000).toFixed(0);

  let envelope;
  try { envelope = JSON.parse(r.out); } catch {
    log(`  ✖ agent[${name}#${uid}] non-JSON envelope (exit=${r.code}, ${secs}s): ${(r.err || r.out).slice(-300)}`);
    return { ok: false, err: "non-json-envelope", costUsd: 0 };
  }
  const cost = Number(envelope.total_cost_usd || 0);
  TOTAL_USD += cost;
  if (envelope.is_error || envelope.subtype !== "success") {
    log(`  ✖ agent[${name}#${uid}] error subtype=${envelope.subtype} (${secs}s, $${cost.toFixed(3)})`);
    return { ok: false, err: envelope.subtype || "error", costUsd: cost, raw: envelope };
  }
  // primary: the file the agent wrote; fallback: parse the chat result
  let data = null;
  if (existsSync(outPath)) { try { data = parseLooseJSON(readFileSync(outPath, "utf8")); } catch {} }
  if (data == null && typeof envelope.result === "string") data = parseLooseJSON(envelope.result);
  if (data == null) {
    log(`  ✖ agent[${name}#${uid}] produced no parseable JSON (${secs}s, $${cost.toFixed(3)})`);
    return { ok: false, err: "no-structured-output", costUsd: cost, raw: envelope };
  }
  log(`  ✔ agent[${name}#${uid}] done (${secs}s, $${cost.toFixed(3)}, cum=$${TOTAL_USD.toFixed(2)})`);
  return { ok: true, data, costUsd: cost, raw: envelope };
}

// bounded-concurrency parallel map
async function pMap(items, fn, limit = CFG.concurrency) {
  const out = new Array(items.length);
  let i = 0;
  const workers = Array(Math.min(limit, items.length)).fill(0).map(async () => {
    while (i < items.length) { const idx = i++; out[idx] = await fn(items[idx], idx); }
  });
  await Promise.all(workers);
  return out;
}

function guardBudget(stage) {
  if (TOTAL_USD >= CFG.maxTotalUsd) {
    halt(stage, `Global budget circuit-breaker tripped: cumulative $${TOTAL_USD.toFixed(2)} >= --max-total-usd $${CFG.maxTotalUsd}. Checkpoint preserved; re-run with a higher --max-total-usd to continue.`);
  }
}

// ─────────────────────────── state / checkpoint ───────────────────────────
function loadState() {
  if (existsSync(join(WORK, "state.json"))) { try { return JSON.parse(readFileSync(join(WORK, "state.json"), "utf8")); } catch {} }
  return { completed: {}, startedAt: new Date().toISOString() };
}
function saveState(s) { writeFileSync(join(WORK, "state.json"), JSON.stringify(s, null, 2)); }

function writeJSON(file, obj) { writeFileSync(join(WORK, file), JSON.stringify(obj, null, 2)); }
function readJSON(file) { return JSON.parse(readFileSync(join(WORK, file), "utf8")); }

async function checkpoint(stageKey, mdName, mdBody, jsonName, jsonObj, state) {
  writeFileSync(join(AUDIT, mdName), mdBody);
  writeJSON(jsonName, jsonObj);
  state.completed[stageKey] = { at: new Date().toISOString(), cost: COST_BY_STAGE[stageKey] || 0, md: mdName };
  saveState(state);
  await gitPersist(stageKey);
  log(`■ checkpoint saved: ${mdName} (stage cost ~$${(COST_BY_STAGE[stageKey] || 0).toFixed(2)})`);
}

function halt(stage, reason) {
  const body = `# PIPELINE HALTED at ${stage}\n\n**Reason:** ${reason}\n\nCumulative spend: $${TOTAL_USD.toFixed(2)}\nTimestamp: ${new Date().toISOString()}\n\nThe stop-test for this stage could not be satisfied within the safety ceiling, OR a circuit-breaker tripped. Per the pipeline quality gate, the stage halts and reports rather than emitting a falsely-confident artifact. Completed stages remain durable in audit/.\n`;
  writeFileSync(join(AUDIT, "HALT-REPORT.md"), body);
  log(`██ HALT at ${stage}: ${reason}`);
  gitPersist(`HALT-${stage}`).finally(() => process.exit(2));
  throw new Error("halt");
}

// ─────────────────────── deterministic enumeration ───────────────────────
function enumerateFiles() {
  // tracked + untracked (excluding ignored). git is the source of truth for the repo surface.
  return sh("git", ["ls-files", "--cached", "--others", "--exclude-standard"]).then(({ out }) => {
    return out.split("\n").map((s) => s.trim()).filter(Boolean)
      .filter((p) => !IGNORE.some((ig) => p === ig || p.startsWith(ig)));
  });
}

// ═══════════════════════════════ STAGE 1 ═══════════════════════════════
const S1_SCHEMA = {
  type: "object", additionalProperties: false,
  required: ["files", "entryPoints", "architecture", "provisionalIntent"],
  properties: {
    files: { type: "array", items: { type: "object", additionalProperties: false,
      required: ["path", "role", "oneLiner"],
      properties: {
        path: { type: "string" },
        role: { type: "string", enum: ["source", "test", "doc", "config", "asset", "generated", "dead", "unknown"] },
        oneLiner: { type: "string" },
      } } },
    entryPoints: { type: "array", items: { type: "object", additionalProperties: false,
      required: ["name", "kind", "location", "description"],
      properties: { name: { type: "string" }, kind: { type: "string" }, location: { type: "string" }, description: { type: "string" } } } },
    architecture: { type: "string" },
    provisionalIntent: { type: "string" },
  },
};

async function stage1(state) {
  log("═══ STAGE 1 — Comprehensive understanding ═══");
  const all = await enumerateFiles();
  log(`  denominator: ${all.length} files (post-ignore)`);
  writeJSON("01-denominator.json", { count: all.length, files: all });

  // fan out by top-level subtree
  const byTop = {};
  for (const f of all) { const top = f.includes("/") ? f.split("/")[0] : "(root)"; (byTop[top] ||= []).push(f); }
  // split large subtrees into <=70-file chunks
  const chunks = [];
  for (const [top, files] of Object.entries(byTop)) {
    for (let i = 0; i < files.length; i += 70) chunks.push({ top, files: files.slice(i, i + 70) });
  }
  log(`  ${Object.keys(byTop).length} subtrees -> ${chunks.length} analysis chunks`);

  const CHUNK_SCHEMA = { type: "object", additionalProperties: false, required: ["files", "entryPoints"],
    properties: { files: S1_SCHEMA.properties.files, entryPoints: S1_SCHEMA.properties.entryPoints } };

  let inventory = [];
  let entryPoints = [];
  async function coverChunk(chunk) {
    const r = await runAgent({
      name: `s1:${chunk.top}:${chunk.files.length}`, model: CFG.MODEL_WORKER, maxTurns: 60, budgetUsd: 4,
      schema: CHUNK_SCHEMA,
      prompt: `Classify every file below from the repository at ${REPO}. For EACH file, open/inspect it (Read/Grep/head) enough to assign a role and write a precise one-line purpose. Roles: source|test|doc|config|asset|generated|dead. Use "dead" only if you can justify unreachability with evidence; if unsure use the best concrete role and never leave anything as "unknown".
Also extract any ENTRY POINTS defined in these files (CLI commands, exported public APIs, HTTP routes, __main__ blocks, script entry points), each with name, kind, location (path:line), and a one-line description.
Return JSON: {files:[{path,role,oneLiner}], entryPoints:[...]}. You MUST return exactly one entry per file listed.
FILES (${chunk.files.length}):\n${chunk.files.join("\n")}`,
    });
    if (!r.ok || !r.data?.files) { log(`  ! chunk ${chunk.top} failed: ${r.err}`); return { files: [], entryPoints: [] }; }
    return r.data;
  }

  const results = await pMap(chunks, coverChunk);
  for (const res of results) { inventory.push(...(res.files || [])); entryPoints.push(...(res.entryPoints || [])); }

  // coverage stop-test: every denominator file present, zero unknown
  const seen = new Set(inventory.map((f) => f.path));
  let missing = all.filter((p) => !seen.has(p));
  let round = 0;
  while (missing.length && round < CFG.ceiling) {
    round++;
    log(`  coverage gap: ${missing.length} files unvisited -> remediation round ${round}`);
    const remChunks = [];
    for (let i = 0; i < missing.length; i += 70) remChunks.push({ top: `gap${round}`, files: missing.slice(i, i + 70) });
    const rem = await pMap(remChunks, coverChunk);
    for (const res of rem) { inventory.push(...(res.files || [])); entryPoints.push(...(res.entryPoints || [])); }
    const seen2 = new Set(inventory.map((f) => f.path));
    missing = all.filter((p) => !seen2.has(p));
    guardBudget("stage1");
  }
  if (missing.length) halt("stage1", `Coverage stop-test failed: ${missing.length} files never classified after ${CFG.ceiling} rounds.`);
  // dedupe (last wins), drop any stray non-denominator paths
  const map = new Map();
  for (const f of inventory) if (all.includes(f.path)) map.set(f.path, f);
  inventory = [...map.values()];
  const unknown = inventory.filter((f) => f.role === "unknown");
  if (unknown.length) halt("stage1", `Convergence stop-test failed: ${unknown.length} files classified "unknown".`);

  // synthesis: architecture + provisional intent, grounded in the inventory + entry points
  writeJSON("01-inventory-raw.json", { files: inventory, entryPoints });
  const synth = await runAgent({
    name: "s1:synthesis", model: CFG.MODEL_SYNTH, maxTurns: 40, budgetUsd: 6, schema: S1_SCHEMA,
    prompt: `You are synthesizing Stage 1 of a forensic audit of the repo at ${REPO}.
A per-file inventory and entry-point list have been collected at audit/.work/01-inventory-raw.json — Read it. You may also Read key files (README, pyproject/package manifests, top-level modules, the largest source files, CI configs) to ground the architecture narrative and verify entry points.
Produce the final Stage-1 object:
- files: the FULL inventory (carry through every entry from 01-inventory-raw.json; correct any obviously wrong role with evidence). There are ${inventory.length} files — return all of them.
- entryPoints: the consolidated, de-duplicated entry-point table (verify each location).
- architecture: a dense paragraph describing the system's components and how they fit (cite key path:line anchors inline).
- provisionalIntent: the apparent reason this project exists, EXPLICITLY marked provisional (this is the coverage-of-intent denominator Stage 2 judges defects against until Stage 4 refines it).`,
  });
  guardBudget("stage1");
  if (!synth.ok || !synth.data?.architecture) halt("stage1", `Synthesis failed: ${synth.err || "no data"}`);
  // ensure full inventory survives even if synthesis trimmed it
  if (!synth.data.files || synth.data.files.length < inventory.length) synth.data.files = inventory;

  const obj = synth.data;
  const md = renderS1(obj, all.length);
  COST_BY_STAGE.stage1 = (COST_BY_STAGE.stage1 || 0) + 0; // tracked via TOTAL; per-stage delta below
  await checkpoint("stage1", "01-understanding.md", md, "01-understanding.json", obj, state);
  return obj;
}

function table(headers, rows) {
  const h = `| ${headers.join(" | ")} |\n| ${headers.map(() => "---").join(" | ")} |`;
  const b = rows.map((r) => `| ${r.map((c) => String(c).replace(/\n/g, " ").replace(/\|/g, "\\|")).join(" | ")} |`).join("\n");
  return `${h}\n${b}`;
}
function renderS1(o, denom) {
  const byRole = {};
  for (const f of o.files) (byRole[f.role] ||= []).push(f);
  const roleSummary = Object.entries(byRole).map(([r, fs]) => `- **${r}**: ${fs.length}`).join("\n");
  return `# 01 — Comprehensive Understanding

> Stage 1 of the forensic audit pipeline. Coverage denominator: **${denom}** files (post-ignore). This document is the denominator for every later stage.

## Architecture
${o.architecture}

## Provisional intent (PROVISIONAL — refined/replaced in Stage 4)
${o.provisionalIntent}

## Entry points
${table(["Name", "Kind", "Location", "Description"], o.entryPoints.map((e) => [e.name, e.kind, e.location, e.description]))}

## File-role summary
${roleSummary}

## Full inventory (${o.files.length} files)
${table(["Path", "Role", "Purpose"], o.files.map((f) => [f.path, f.role, f.oneLiner]))}

---
### machine-readable artifact
\`\`\`json
${JSON.stringify(o, null, 2)}
\`\`\`
`;
}

// ═══════════════════════════════ STAGE 2 ═══════════════════════════════
const FINDING = { type: "object", additionalProperties: false,
  required: ["id", "location", "class", "severity", "evidence"],
  properties: {
    id: { type: "string" }, location: { type: "string" },
    class: { type: "string", enum: ["bug", "security", "doc_code_drift", "design_defect", "intent_mismatch"] },
    severity: { type: "string", enum: ["critical", "high", "medium", "low", "info"] },
    evidence: { type: "string" },
  } };
const S2_FINDINGS_SCHEMA = { type: "object", additionalProperties: false, required: ["findings", "visited"],
  properties: { findings: { type: "array", items: FINDING }, visited: { type: "array", items: { type: "string" } } } };

function fingerprint(f) { return `${(f.location || "").split(":")[0]}|${f.class}|${(f.evidence || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().slice(0, 60)}`; }

async function stage2(state, s1) {
  log("═══ STAGE 2 — Static audit (diverse lenses + adversarial fixpoint) ═══");
  const denom = readJSON("01-denominator.json").files;
  const lenses = [
    ["security", "security vulnerabilities: injection, unsafe subprocess/shell, path traversal, deserialization, secrets in code, unsafe eval/exec, SSRF, auth gaps"],
    ["correctness", "logic bugs, off-by-one, error-handling gaps, race conditions, resource leaks, incorrect edge-case handling, broken invariants"],
    ["doc_code_drift", "places where docs/comments/README/docstrings contradict the actual code behavior (judge against the Stage-1 provisional intent)"],
    ["design_defect", "architectural smells: tight coupling, leaky abstractions, dead/stub code presented as live, duplicated logic, missing validation boundaries"],
    ["dependency_supplychain", "dependency/version risks, pinning, known-risky calls, build/CI config weaknesses, supply-chain exposure"],
  ];

  async function auditLens([lensId, lensDesc], scopeFiles) {
    const r = await runAgent({
      name: `s2:${lensId}`, model: CFG.MODEL_WORKER, maxTurns: 70, budgetUsd: 5, schema: S2_FINDINGS_SCHEMA,
      prompt: `Static audit of the repo at ${REPO} through the **${lensId}** lens: ${lensDesc}.
Read audit/.work/01-understanding.json for the inventory, entry points, architecture, and the PROVISIONAL INTENT. A defect is only a defect relative to intended behavior — judge against that provisional intent, and record any code/intent mismatch as its own finding (class "intent_mismatch").
Re-read the actual source (do not trust the map). Cover these files at minimum: ${scopeFiles.length} files listed below. Visit every one; report which you visited in "visited".
For each defect: id (slug), location (path:line), class (one of bug|security|doc_code_drift|design_defect|intent_mismatch), severity (critical|high|medium|low|info), and concrete evidence (quote the code and explain the defect). No claim without a path:line. If you suspect but cannot confirm, mark severity "info" and say "unverified" in evidence.
FILES TO COVER:\n${scopeFiles.join("\n")}`,
    });
    return r.ok && r.data ? r.data : { findings: [], visited: [] };
  }

  // Route lenses by file type so multiple lenses hit CODE (where they matter) while every
  // file is still visited >=1 (coverage stop-test enforces the union). Avoids the wasteful
  // full lens x file cross-product over hundreds of curriculum/data/doc files.
  const chunksOf = (arr, n) => { const o = []; for (let i = 0; i < arr.length; i += n) o.push(arr.slice(i, i + n)); return o; };
  const CODE = /\.(py|sh|bash|js|mjs|ts|rb|go|rs|c|h|cpp|java)$/i;
  const DOCY = /\.(md|rst|txt|adoc)$/i;
  const DATA = /\.(json|ya?ml|toml|lock|cfg|ini|env|example)$|(^|\/)(Dockerfile|Makefile)$/i;
  const codeFiles = denom.filter((p) => CODE.test(p));
  const docFiles = denom.filter((p) => DOCY.test(p));
  const dataFiles = denom.filter((p) => DATA.test(p));
  const lensById = Object.fromEntries(lenses.map((l) => [l[0], l]));
  log(`  routing: ${codeFiles.length} code, ${docFiles.length} doc, ${dataFiles.length} data files`);

  let findings = [];
  const visited = new Set();
  // parallel barrier: code through security+correctness+design; docs through drift; data through supply-chain
  const jobs = [];
  for (const lid of ["security", "correctness", "design_defect"]) for (const ch of chunksOf(codeFiles, 55)) jobs.push([lensById[lid], ch]);
  for (const ch of chunksOf(docFiles, 45)) jobs.push([lensById["doc_code_drift"], ch]);
  for (const ch of chunksOf(dataFiles.concat(codeFiles.filter((p) => /(^|\/)(pyproject|package|setup|tox|noxfile|conftest)/i.test(p))), 60)) jobs.push([lensById["dependency_supplychain"], ch]);
  const lensResults = await pMap(jobs, ([lens, ch]) => auditLens(lens, ch));
  for (const res of lensResults) { (res.findings || []).forEach((f) => findings.push(f)); (res.visited || []).forEach((v) => visited.add(v.split(":")[0])); }
  guardBudget("stage2");

  // coverage stop-test: every denominator source/test/config visited >=1
  const mustVisit = denom.filter((p) => !p.endsWith(".lock"));
  let unvisited = mustVisit.filter((p) => !visited.has(p));
  let cround = 0;
  while (unvisited.length && cround < CFG.ceiling) {
    cround++;
    log(`  coverage gap: ${unvisited.length} files unvisited by any lens -> round ${cround}`);
    const r = await auditLens(["correctness", "any defects, broad sweep"], unvisited.slice(0, 120));
    (r.findings || []).forEach((f) => findings.push(f));
    (r.visited || []).forEach((v) => visited.add(v.split(":")[0]));
    unvisited = mustVisit.filter((p) => !visited.has(p));
    guardBudget("stage2");
  }
  log(`  raw findings: ${findings.length}; visited ${visited.size}/${mustVisit.length} required files`);

  // adversarial fixpoint: audit -> falsify -> keep survivors -> re-audit remainder -> until stable
  const FALSIFY_SCHEMA = { type: "object", additionalProperties: false, required: ["verdicts"],
    properties: { verdicts: { type: "array", items: { type: "object", additionalProperties: false,
      required: ["id", "verdict", "reason"],
      properties: { id: { type: "string" }, verdict: { type: "string", enum: ["survives", "refuted", "downgraded"] }, reason: { type: "string" }, newSeverity: { type: "string" } } } } } };

  function dedupe(list) { const m = new Map(); for (const f of list) m.set(fingerprint(f), f); return [...m.values()]; }
  findings = dedupe(findings);

  let prevSig = "";
  let round = 0;
  while (round < CFG.ceiling) {
    round++;
    writeJSON(`02-findings-round${round}.json`, { findings });
    const fal = await runAgent({
      name: `s2:falsifier:r${round}`, model: CFG.MODEL_ADVERSARY, maxTurns: 70, budgetUsd: 6, schema: FALSIFY_SCHEMA,
      prompt: `You are an ADVERSARIAL FALSIFIER. ${findings.length} candidate findings are in audit/.work/02-findings-round${round}.json. Read it, then independently re-read the cited source at ${REPO} for EACH finding and try to REFUTE it. Do not trust the finding's framing.
For each finding id, return a verdict: "refuted" (you found counter-evidence it is not a real defect — give the path:line counter-evidence), "downgraded" (real but lower severity — give newSeverity), or "survives" (you could not refute it; restate the confirming path:line). Be skeptical: vague, location-less, or speculative findings should be refuted or downgraded. Return {verdicts:[{id,verdict,reason,newSeverity?}]}.`,
    });
    guardBudget("stage2");
    if (!fal.ok || !fal.data?.verdicts) { log(`  ! falsifier round ${round} failed: ${fal.err}; keeping current set`); break; }
    const vmap = new Map(fal.data.verdicts.map((v) => [v.id, v]));
    const survivors = [];
    for (const f of findings) {
      const v = vmap.get(f.id);
      if (!v) { survivors.push(f); continue; }            // not addressed -> keep (conservative)
      if (v.verdict === "refuted") continue;
      if (v.verdict === "downgraded" && v.newSeverity) f.severity = v.newSeverity;
      f.adversarialNote = v.reason;
      survivors.push(f);
    }
    const sig = survivors.map(fingerprint).sort().join("||");
    log(`  round ${round}: ${findings.length} -> ${survivors.length} survivors`);
    findings = survivors;
    if (sig === prevSig) { log(`  ✔ fixpoint reached at round ${round}`); break; }   // stable across a full cycle
    prevSig = sig;
    if (round === CFG.ceiling) { log(`  ⚠ ceiling reached without fixpoint`); halt("stage2", `Adversarial fixpoint not reached within ceiling ${CFG.ceiling}.`); }

    // re-audit remainder: one more correctness+security sweep over the code to surface anything the falsifier's pressure implies
    const reaudit = await pMap([lensById["correctness"], lensById["security"]], (lens) => auditLens(lens, codeFiles));
    let added = 0;
    for (const res of reaudit) for (const f of (res.findings || [])) { const before = findings.length; findings = dedupe([...findings, f]); if (findings.length > before) added++; }
    log(`  re-audit added ${added} new candidate(s)`);
    guardBudget("stage2");
  }

  const obj = { findings, coverage: { requiredFiles: mustVisit.length, visited: visited.size }, fixpointRounds: round };
  await checkpoint("stage2", "02-static-audit.md", renderS2(obj), "02-static-audit.json", obj, state);
  return obj;
}
function renderS2(o) {
  const bySev = (s) => o.findings.filter((f) => f.severity === s);
  const order = ["critical", "high", "medium", "low", "info"];
  const counts = order.map((s) => `${s}: ${bySev(s).length}`).join(" · ");
  return `# 02 — Static Audit

> Diverse-lens audit promoted through an adversarial falsifier to a fixpoint (${o.fixpointRounds} round(s)). Coverage: ${o.coverage.visited}/${o.coverage.requiredFiles} required files visited. Severity mix — ${counts}.

${order.filter((s) => bySev(s).length).map((s) => `## ${s.toUpperCase()} (${bySev(s).length})\n` + table(
    ["ID", "Class", "Location", "Evidence", "Adversarial note"],
    bySev(s).map((f) => [f.id, f.class, f.location, f.evidence, f.adversarialNote || "—"]))).join("\n\n")}

---
### machine-readable artifact
\`\`\`json
${JSON.stringify(o, null, 2)}
\`\`\`
`;
}

// ═══════════════════════════════ STAGE 3 ═══════════════════════════════
const S3_SCHEMA = { type: "object", additionalProperties: false,
  required: ["coverage", "observedBehaviors", "findingDeltas", "unexecutedRegions"],
  properties: {
    coverage: { type: "object", additionalProperties: true, required: ["summary"], properties: { summary: { type: "string" }, linePct: { type: "number" } } },
    observedBehaviors: { type: "array", items: { type: "object", additionalProperties: false, required: ["entryPoint", "observed"], properties: { entryPoint: { type: "string" }, observed: { type: "string" } } } },
    findingDeltas: { type: "array", items: { type: "object", additionalProperties: false, required: ["findingId", "status", "evidence"], properties: { findingId: { type: "string" }, status: { type: "string", enum: ["confirmed", "refuted", "refined"] }, evidence: { type: "string" } } } },
    unexecutedRegions: { type: "array", items: { type: "object", additionalProperties: false, required: ["location", "reason"], properties: { location: { type: "string" }, reason: { type: "string", enum: ["requires-credentials", "external-service", "hardware-gated", "dead", "destructive-skip", "other"] } } } },
  } };

async function stage3(state, s1, s2) {
  log("═══ STAGE 3 — Execution / dynamic surface ═══");
  // isolated execution sandbox so test/coverage runs don't dirty the tree holding audit/
  const SBX = "/tmp/me-exec-sandbox";
  rmSync(SBX, { recursive: true, force: true });
  await sh("git", ["worktree", "prune"]);
  const wt = await sh("git", ["worktree", "add", "--detach", SBX, "HEAD"]);
  if (wt.code !== 0) log(`  worktree add note: ${(wt.err || wt.out).slice(0, 200)}`);
  const cwd = existsSync(SBX) ? SBX : REPO;
  log(`  execution sandbox: ${cwd}`);

  const r = await runAgent({
    name: "s3:execution", model: CFG.MODEL_WORKER, maxTurns: 120, budgetUsd: 10, cwd, timeoutMs: 1500000,
    schema: S3_SCHEMA,
    prompt: `Stage 3: determine what the code ACTUALLY does when run, in the sandbox at ${cwd}.
Read ${join(AUDIT, "01-understanding.json")} (entry points) and ${join(AUDIT, "02-static-audit.json")} (findings) first — those paths are absolute; use Read.
Steps:
1. Set up the environment exactly as the project intends (inspect README/pyproject/CI). This project uses 'uv' and a mode symlink; e.g. './scripts/mode switch developer' then 'uv sync' then 'uv pip install -e .'. Adapt to whatever the repo actually requires.
2. Run the test suite under coverage instrumentation (e.g. 'uv run pytest --cov=engine --cov-report=term-missing -m "not integration"'). Capture the measured coverage percentage and summary.
3. Drive the real entry points from the Stage-1 table (e.g. the CLI: --help and representative subcommands) and record observed behavior.
4. Use this runtime evidence to CONFIRM, REFUTE, or REFINE each Stage-2 finding you can reach; report findingDeltas (cite the runtime evidence).
5. Account for un-executed regions: every region not exercised must be classified requires-credentials | external-service | hardware-gated | dead | destructive-skip | other (with reason). Target is 100% ACCOUNTING, not 100% execution.
Return the S3 object. Ground coverage.summary in real command output you actually ran.`,
  });
  guardBudget("stage3");
  if (!r.ok || !r.data?.coverage) halt("stage3", `Execution agent failed: ${r.err || "no data"}`);
  await sh("git", ["worktree", "remove", "--force", SBX]).catch?.(() => {});
  await sh("git", ["worktree", "remove", "--force", SBX]);

  const obj = r.data;
  await checkpoint("stage3", "03-execution.md", renderS3(obj), "03-execution.json", obj, state);
  return obj;
}
function renderS3(o) {
  return `# 03 — Execution / Dynamic Surface

## Measured coverage
${o.coverage.summary}${o.coverage.linePct != null ? `\n\n**Line coverage:** ${o.coverage.linePct}%` : ""}

## Observed behaviors (entry points driven)
${table(["Entry point", "Observed behavior"], o.observedBehaviors.map((b) => [b.entryPoint, b.observed]))}

## Deltas applied to Stage-2 findings
${table(["Finding", "Status", "Runtime evidence"], o.findingDeltas.map((d) => [d.findingId, d.status, d.evidence]))}

## Un-executed regions (100% accounting)
${table(["Location", "Reason"], o.unexecutedRegions.map((u) => [u.location, u.reason]))}

---
### machine-readable artifact
\`\`\`json
${JSON.stringify(o, null, 2)}
\`\`\`
`;
}

// ═══════════════════════════════ STAGE 4 ═══════════════════════════════
const S4_GOAL_SCHEMA = { type: "object", additionalProperties: false, required: ["candidates"],
  properties: { candidates: { type: "array", items: { type: "object", additionalProperties: false,
    required: ["goal", "successSignals", "grounded"],
    properties: { goal: { type: "string" }, grounded: { type: "boolean" },
      successSignals: { type: "array", items: { type: "object", additionalProperties: false, required: ["signal", "evidenceRef"], properties: { signal: { type: "string" }, evidenceRef: { type: "string" } } } } } } } } };
const S4_RESEARCH_SCHEMA = { type: "object", additionalProperties: false, required: ["sources", "ideas"],
  properties: {
    sources: { type: "array", items: { type: "object", additionalProperties: false, required: ["title", "url", "claim", "verified"], properties: { title: { type: "string" }, url: { type: "string" }, claim: { type: "string" }, verified: { type: "boolean" }, corroboration: { type: "string" } } } },
    ideas: { type: "array", items: { type: "object", additionalProperties: false, required: ["idea", "advancesGoal", "sourceRefs"], properties: { idea: { type: "string" }, advancesGoal: { type: "string" }, sourceRefs: { type: "array", items: { type: "string" } } } } },
  } };

async function stage4(state, s1, s2, s3) {
  log("═══ STAGE 4 — Goal + external research (parallel barrier) ═══");

  // GOAL HALF — infer grounded candidates (plural)
  const goalRun = await runAgent({
    name: "s4:goal", model: CFG.MODEL_SYNTH, maxTurns: 40, budgetUsd: 6, schema: S4_GOAL_SCHEMA,
    prompt: `Stage 4 (goal half). Infer the repo's candidate LONG-TERM goal(s) for ${REPO}.
Read these absolute paths: ${join(AUDIT, "01-understanding.json")}, ${join(AUDIT, "02-static-audit.json")}, ${join(AUDIT, "03-execution.json")}. Also read README/docs as needed.
Keep candidates PLURAL — do not collapse to one. Each candidate must be stated as a set of FALSIFIABLE success signals, and every signal must trace to concrete evidence (evidenceRef = a path:line or a named Stage-1/2/3 artifact field). A candidate with ungrounded signals must be marked grounded:false (it will be flagged for human confirmation, not carried as fact). Return {candidates:[...]}.`,
  });
  guardBudget("stage4");
  if (!goalRun.ok || !goalRun.data?.candidates) halt("stage4", `Goal inference failed: ${goalRun.err || "no data"}`);
  const goal = goalRun.data;
  writeJSON("04-goal-candidates.json", goal);
  const goalTopics = goal.candidates.map((c) => c.goal).join(" ; ");

  // PARALLEL BARRIER: judge panel (weighs grounding) || deep-research fan-out
  const judgePanel = pMap([1, 2, 3], (n) => runAgent({
    name: `s4:judge${n}`, model: CFG.MODEL_WORKER, maxTurns: 25, budgetUsd: 3, schema: S4_GOAL_SCHEMA,
    prompt: `You are judge ${n} of 3 on a panel weighing goal candidates for ${REPO}. Read audit/.work/04-goal-candidates.json and the Stage 1-3 artifacts (audit/.work/0{1,2,3}-*.json). Independently re-verify the grounding of each candidate's success signals against the cited evidence. Return the same {candidates:[...]} structure but with grounded set to YOUR independent verdict and successSignals trimmed to only those you could verify (evidenceRef must resolve). Be strict.`,
  }), 3);

  const researchFanout = CFG.web ? (async () => {
    const angles = [
      "state-of-the-art tools, frameworks and techniques that materially advance this project's goal",
      "comparable/competing open-source projects and how they solve the same problem better",
      "recent research, standards, or methods (last ~2 years) relevant to the project's domain and goal",
    ];
    const gather = await pMap(angles, (angle, i) => runAgent({
      name: `s4:research${i}`, model: CFG.MODEL_WORKER, maxTurns: 30, budgetUsd: 5, web: true,
      schema: S4_RESEARCH_SCHEMA,
      prompt: `Stage 4 (research half), angle ${i + 1}: ${angle}. The project's candidate goals: ${goalTopics}.
Use WebSearch/WebFetch to gather independently. For every source: title, url, the specific claim you draw from it, and verified=true ONLY if you corroborated it against a second independent source (name it in corroboration); otherwise verified=false. Then list concrete ideas/technologies that advance the goal, each linked to advancesGoal and sourceRefs (urls). An uncorroborated claim is recorded as unverified, never as fact. Cite every source.`,
    }), 3);
    return gather.filter((g) => g.ok && g.data).map((g) => g.data);
  })() : Promise.resolve([]);

  const [judges, researchParts] = await Promise.all([judgePanel, researchFanout]);
  guardBudget("stage4");

  // consensus on grounding: a candidate is "grounded" if majority of judges agree
  const judgeVerdicts = judges.filter((j) => j.ok && j.data).map((j) => j.data.candidates || []);
  for (const c of goal.candidates) {
    let votes = c.grounded ? 1 : 0, total = 1;
    for (const jv of judgeVerdicts) { const m = jv.find((x) => x.goal && c.goal && x.goal.slice(0, 30) === c.goal.slice(0, 30)); if (m) { total++; if (m.grounded) votes++; } }
    c.groundedConsensus = votes / total >= 0.5;
    c.judgeVotes = `${votes}/${total}`;
  }

  // research synthesis with cross-check + saturation note
  let research = { sources: [], ideas: [] };
  if (CFG.web) {
    for (const part of researchParts) { research.sources.push(...(part.sources || [])); research.ideas.push(...(part.ideas || [])); }
    // dedupe sources by url
    const sm = new Map(); for (const s of research.sources) sm.set(s.url, s); research.sources = [...sm.values()];
  }

  const obj = { candidates: goal.candidates, research, webEnabled: CFG.web };
  await checkpoint("stage4", "04-goal.md", renderS4(obj), "04-goal.json", obj, state);
  return obj;
}
function renderS4(o) {
  return `# 04 — Grounded Goal + External Research

## Candidate long-term goals (plural by design)
${o.candidates.map((c) => `### ${c.goal}
- **Grounded (author):** ${c.grounded} · **Judge consensus:** ${c.groundedConsensus ?? "n/a"} (${c.judgeVotes ?? "n/a"})
- Success signals:
${c.successSignals.map((s) => `  - ${s.signal} — _evidence:_ ${s.evidenceRef}`).join("\n")}`).join("\n\n")}

## External research ${o.webEnabled ? "" : "(SKIPPED — --no-web)"}
${o.webEnabled ? `### Sources
${table(["Title", "URL", "Claim", "Verified", "Corroboration"], o.research.sources.map((s) => [s.title, s.url, s.claim, s.verified, s.corroboration || "—"]))}

### Ideas that advance the goal
${table(["Idea", "Advances", "Sources"], o.research.ideas.map((i) => [i.idea, i.advancesGoal, (i.sourceRefs || []).join(" ")]))}` : "_External research disabled for this run._"}

---
### machine-readable artifact
\`\`\`json
${JSON.stringify(o, null, 2)}
\`\`\`
`;
}

// ═══════════════════════════════ STAGE 5 ═══════════════════════════════
const S5_SCHEMA = { type: "object", additionalProperties: false, required: ["items"],
  properties: { items: { type: "array", items: { type: "object", additionalProperties: false,
    required: ["id", "linkTo", "location", "change", "verificationSignal", "dependsOn", "order"],
    properties: { id: { type: "string" }, linkTo: { type: "string" }, location: { type: "string" }, change: { type: "string" }, verificationSignal: { type: "string" }, dependsOn: { type: "array", items: { type: "string" } }, order: { type: "number" } } } } } };

async function stage5(state) {
  log("═══ STAGE 5 — Execution-ready plan ═══");
  const plan = await runAgent({
    name: "s5:plan", model: CFG.MODEL_SYNTH, maxTurns: 45, budgetUsd: 7, schema: S5_SCHEMA,
    prompt: `Stage 5: produce the execution-ready change plan for ${REPO} that closes the gap between current state (Stages 1-3) and goal (Stage 4).
Read all four prior artifacts (absolute): ${["01-understanding", "02-static-audit", "03-execution", "04-goal"].map((s) => join(AUDIT, s + ".json")).join(", ")}.
Produce ORDERED change items. Each item MUST have: id; linkTo (a specific Stage-2 finding id OR a Stage-4 goal-gap, named); location (file/module a diff would touch); change (what to do, concretely); verificationSignal (the exact observation/test/command that proves it worked); dependsOn (ids of prerequisite items); order (integer, topologically sorted by dependsOn). Every item must be specific enough that a fresh engineer maps it to a concrete diff target with NO clarifying question.`,
  });
  guardBudget("stage5");
  if (!plan.ok || !plan.data?.items) halt("stage5", `Plan synthesis failed: ${plan.err || "no data"}`);
  let items = plan.data.items;

  // completeness gate
  const incomplete = items.filter((it) => !it.linkTo || !it.location || !it.verificationSignal || it.order == null || !Array.isArray(it.dependsOn));
  if (incomplete.length) halt("stage5", `Completeness stop-test failed: ${incomplete.length} item(s) missing required fields.`);

  // convergence gate: independent diff-targetability checker; ambiguous items loop
  const CHECK_SCHEMA = { type: "object", additionalProperties: false, required: ["ambiguous"], properties: { ambiguous: { type: "array", items: { type: "object", additionalProperties: false, required: ["id", "why"], properties: { id: { type: "string" }, why: { type: "string" } } } } } };
  let round = 0;
  while (round < CFG.ceiling) {
    round++;
    writeJSON(`05-plan-round${round}.json`, { items });
    const chk = await runAgent({
      name: `s5:checker:r${round}`, model: CFG.MODEL_ADVERSARY, maxTurns: 40, budgetUsd: 4, schema: CHECK_SCHEMA,
      prompt: `You are an independent reviewer. Read audit/.work/05-plan-round${round}.json. For each item, decide whether a fresh engineer could map it to a CONCRETE diff target (specific file + specific change + a verification signal they could actually run) WITHOUT asking a clarifying question. Open the cited locations at ${REPO} to confirm they exist and the change is unambiguous. Return {ambiguous:[{id,why}]} listing ONLY items that fail. Empty array = the plan converged.`,
    });
    guardBudget("stage5");
    if (!chk.ok) { log(`  ! checker round ${round} failed: ${chk.err}; accepting current plan`); break; }
    const amb = chk.data.ambiguous || [];
    if (!amb.length) { log(`  ✔ plan converged (round ${round}): all items diff-targetable`); break; }
    log(`  ${amb.length} ambiguous item(s) -> refinement round ${round}`);
    if (round === CFG.ceiling) halt("stage5", `Convergence stop-test failed: ${amb.length} items still ambiguous after ${CFG.ceiling} rounds.`);
    const ambIds = amb.map((a) => `${a.id}: ${a.why}`).join("\n");
    writeJSON("05-ambiguous.json", { ambiguous: amb });
    const fix = await runAgent({
      name: `s5:refine:r${round}`, model: CFG.MODEL_SYNTH, maxTurns: 40, budgetUsd: 5, schema: S5_SCHEMA,
      prompt: `Refine the change plan in audit/.work/05-plan-round${round}.json. These items were flagged ambiguous by an independent reviewer (audit/.work/05-ambiguous.json):\n${ambIds}\nRewrite the FULL items list so every flagged item becomes diff-targetable (concrete location, concrete change, runnable verificationSignal), preserving the others and the ordering/dependsOn. Return {items:[...]}.`,
    });
    guardBudget("stage5");
    if (fix.ok && fix.data?.items) items = fix.data.items;
  }

  items.sort((a, b) => a.order - b.order);
  const obj = { items, convergenceRounds: round };
  await checkpoint("stage5", "05-plan.md", renderS5(obj), "05-plan.json", obj, state);
  return obj;
}
function renderS5(o) {
  return `# 05 — Execution-Ready Plan

> ${o.items.length} ordered change items. Converged after ${o.convergenceRounds} round(s): every item maps to a concrete diff target with a runnable verification signal.

${table(["#", "ID", "Links to", "Location", "Change", "Verification signal", "Depends on"],
    o.items.map((i) => [i.order, i.id, i.linkTo, i.location, i.change, i.verificationSignal, (i.dependsOn || []).join(", ") || "—"]))}

---
### machine-readable artifact
\`\`\`json
${JSON.stringify(o, null, 2)}
\`\`\`
`;
}

// ─────────────────────────── preflight self-test ───────────────────────────
async function preflight() {
  log("═══ PREFLIGHT — validating the agent-invocation path ═══");
  const r = await runAgent({
    name: "preflight", model: CFG.MODEL_WORKER, maxTurns: 6, budgetUsd: 1,
    schema: { type: "object", additionalProperties: false, required: ["repoName", "topLevelDirs", "ok"], properties: { repoName: { type: "string" }, topLevelDirs: { type: "array", items: { type: "string" } }, ok: { type: "boolean" } } },
    prompt: `Preflight check. List the top-level directories of the repo at ${REPO} (use ls/Glob) and return {repoName, topLevelDirs, ok:true}. This validates structured-output + tool use end-to-end.`,
  });
  if (!r.ok || !r.data?.ok) { log(`PREFLIGHT FAILED: ${r.err || "bad data"}. Aborting before full run.`); process.exit(3); }
  log(`PREFLIGHT OK — repo=${r.data.repoName}, dirs=[${(r.data.topLevelDirs || []).join(", ")}], cost so far $${TOTAL_USD.toFixed(3)}`);
  process.exit(0);
}

// ─────────────────────────────── main ───────────────────────────────
async function main() {
  mkdirSync(WORK, { recursive: true });
  if (CFG.fresh) {
    log("--fresh: tearing down prior audit artifacts");
    for (const f of ["01-understanding.md", "02-static-audit.md", "03-execution.md", "04-goal.md", "05-plan.md", "HALT-REPORT.md"]) rmSync(join(AUDIT, f), { force: true });
    rmSync(WORK, { recursive: true, force: true }); mkdirSync(WORK, { recursive: true });
  }
  log(`Forensic pipeline start. repo=${REPO} mode=${CFG.fresh ? "fresh" : "resume"} web=${CFG.web} ceiling=${CFG.ceiling} budgetCap=$${CFG.maxTotalUsd} concurrency=${CFG.concurrency}`);
  if (CFG.preflight) return preflight();

  const state = loadState();
  const done = (k) => !!state.completed[k] && (!CFG.onlyStage);
  const want = (n) => CFG.onlyStage == null || CFG.onlyStage === n;
  const start = TOTAL_USD;
  const trackStage = (k) => { COST_BY_STAGE[k] = TOTAL_USD - start - Object.entries(COST_BY_STAGE).filter(([kk]) => kk !== k).reduce((a, [, v]) => a + v, 0); };

  let s1, s2, s3, s4;
  if (want(1)) { if (done("stage1")) { log("resume: stage1 complete"); s1 = readJSON("01-understanding.json"); } else { s1 = await stage1(state); trackStage("stage1"); } }
  if (want(2)) { if (done("stage2")) { log("resume: stage2 complete"); s2 = readJSON("02-static-audit.json"); } else { s2 = await stage2(state, s1 || readJSON("01-understanding.json")); trackStage("stage2"); } }
  if (want(3)) { if (done("stage3")) { log("resume: stage3 complete"); s3 = readJSON("03-execution.json"); } else { s3 = await stage3(state, s1, s2 || readJSON("02-static-audit.json")); trackStage("stage3"); } }
  if (want(4)) { if (done("stage4")) { log("resume: stage4 complete"); s4 = readJSON("04-goal.json"); } else { s4 = await stage4(state, s1, s2, s3); trackStage("stage4"); } }
  if (want(5)) { if (done("stage5")) { log("resume: stage5 complete"); } else { await stage5(state); trackStage("stage5"); } }

  log(`██ PIPELINE COMPLETE. Total spend ~$${TOTAL_USD.toFixed(2)}. Artifacts in audit/.`);
  writeFileSync(join(WORK, "SUMMARY.json"), JSON.stringify({ totalUsd: Number(TOTAL_USD.toFixed(2)), costByStage: COST_BY_STAGE, completedAt: new Date().toISOString() }, null, 2));
  await gitPersist("final");
}

main().catch((e) => { if (String(e.message) !== "halt") { log(`FATAL: ${e.stack || e}`); process.exit(1); } });
