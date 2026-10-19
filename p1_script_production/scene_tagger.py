"""Scene tagger v1 (P1, LLM block).

Reads each scene's action text and rates what it implies for production cost.
Speaking characters, location, INT/EXT and day/night come from the rules-based
Scene parser (free and exact), so the model only rates things that need reading.

Commands (run from this folder):
  python scene_tagger.py plan   [--in_price X --out_price Y]
  python scene_tagger.py run    --set pilot5|pilot200|all [--runs 3] [--model M] [--workers 4] [--fake] [--max_chunks N]
  python scene_tagger.py report --set pilot5|pilot200|all [--runs 3] [--model M] [--fake]

--fake runs the whole pipeline with invented answers: no key, no cost. Use it to test.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from scene_parser import parse_script, clean

TAGGER_VERSION = "scene_tagger_v1"
PROMPT_VERSION = "scene_tagger_prompt_v1"
DEFAULT_MODEL = "claude-haiku-5-5"
MAX_SCENES = 25          # scenes per call
MAX_CHARS = 28000        # characters of scene text per call (about 7,000 tokens)
DESC_HEAD, DESC_TAIL = 2200, 800   # long action text keeps its start and its end
OUT_TOK_PER_SCENE = 34   # rough size of one scene's answer, for the cost estimate only

# field code: (column name, highest level)
FIELDS = {
    "ex": ("extras", 3),
    "st": ("stunts", 3),
    "vfx": ("vfx", 3),
    "set": ("set_complexity", 3),
    "cm": ("costume_makeup", 3),
    "mu": ("music", 2),
}
VOCAB = {
    "water": "on, in or under water, or boats under way at sea",
    "weather": "rain, snow, storm, fog or strong wind that must be made or shot in",
    "fire_explosions": "fire, pyrotechnics or explosions",
    "vehicles": "moving vehicles on screen: driving, trains, planes, boats under way",
    "aerial": "helicopters, planes in flight, or aerial photography",
    "animals": "live animals that act or must be handled",
    "children": "child actors with real action or lines",
    "firearms": "guns fired, not just present",
    "period_setting": "historical period needing period props, vehicles or streets",
    "gore_or_prosthetics": "blood, gore, wounds or prosthetic effects",
    "remote_location": "clearly remote or exotic place that implies travel: desert, jungle, mountains, abroad",
    "wire_or_rig": "wire work, zero gravity, or special camera rigs",
}

SYSTEM_TEMPLATE = """You are a film production breakdown assistant, doing the first pass a line producer does on a script. You will be given consecutive scenes from a screenplay. For each scene, rate what its ACTION TEXT implies for the cost of shooting it. Rate what must be filmed, not how good the scene is. Judge only from the text shown (dialogue text is not shown). If the text does not mention it, rate 0. Do not use outside knowledge of the film. When unsure between two levels, choose the lower one.

Fields (integers):
ex = extras, non-speaking background people. 0 none, or only the named characters. 1 a few (about 1 to 9). 2 a crowd (about 10 to 50): restaurant, bar, street, office floor. 3 a large crowd (50+): stadium, battle, riot, parade.
st = stunts and physical danger. 0 none. 1 minor: a punch, a shove, a simple fall, ordinary driving. 2 real stunt work: choreographed fight, car crash, fall from height, fire gag. 3 major set-piece: chase, battle, building collapse, big crash sequence.
vfx = visual effects. 0 none. 1 minor: screens or displays, simple wire or sky replacement, blood. 2 notable: explosions, a creature or ghost shown briefly, digital environment extension. 3 heavy: fully computer-made creatures, characters or worlds, or large destruction.
set = set and art department burden. 0 ordinary real location or plain room. 1 dressed location with specific props. 2 elaborate, period or large set: palace, ship, courtroom, lab, historic street. 3 major build or fantasy or sci-fi world: spaceship, alien planet, medieval city.
cm = costume and makeup. 0 ordinary modern everyday clothes. 1 uniforms or character-specific looks. 2 period, military or heavy costume, noticeable aging or wounds. 3 prosthetics, creature or fantasy makeup, or elaborate costume for many people.
mu = music. 0 none. 1 a song, band or radio music is mentioned or plays during the scene. 2 an on-screen performance or musical number is central.
c = special conditions, a list. Use only these codes, and only when the text clearly shows them:
@@CODES@@

Answer with ONLY a JSON array, no other text: exactly one object per scene, in the same order, using the scene numbers given. Each object looks like:
{"n": 12, "ex": 0, "st": 0, "vfx": 0, "set": 0, "cm": 0, "mu": 0, "c": []}"""
SYSTEM = SYSTEM_TEMPLATE.replace("@@CODES@@", "\n".join("%s = %s" % (k, v) for k, v in VOCAB.items()))


class Fatal(Exception):
    pass


def paths():
    here = os.path.dirname(os.path.abspath(__file__))
    raw = os.path.join(here, "..", "..", "raw_data")
    tdir = os.path.join(raw, "tagger")
    os.makedirs(tdir, exist_ok=True)
    return here, raw, tdir


# ---------- reading scripts ----------
def tag_scenes(script_text):
    """Scene dicts for the tagger. Scenes with no action text and no speakers are skipped."""
    out = []
    for s in parse_script(script_text):
        descs = re.findall(r"<scene_description>(.*?)</scene_description>", s["text"], re.S)
        desc = clean(" ".join(descs))
        if not desc and not s["speakers"]:
            continue
        if len(desc) > DESC_HEAD + DESC_TAIL:
            desc = desc[:DESC_HEAD] + " [...] " + desc[-DESC_TAIL:]
        out.append({
            "n": s["scene_no"], "heading": s["heading"] or "(no heading)",
            "speakers": sorted(s["speakers"])[:12], "n_speakers": s["n_speakers"],
            "dlines": s["dialogue_lines"], "desc": desc,
            "int_ext": s["int_ext"], "location": s["location"], "day_night": s["day_night"],
        })
    return out


def iter_scripts(raw, wanted=None):
    with open(os.path.join(raw, "moviesum_clean.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if wanted is not None and d["imdb_id"] not in wanted:
                continue
            yield {"imdb_id": d["imdb_id"], "movie_name": d["movie_name"], "year": d["year"],
                   "scenes": tag_scenes(d["script"])}


def render_scene(sc):
    sp = ", ".join(sc["speakers"]) if sc["speakers"] else "none"
    return "[SCENE %d] %s\nSpeaking characters (%d, %d dialogue lines): %s\nAction: %s\n" % (
        sc["n"], sc["heading"], sc["n_speakers"], sc["dlines"], sp, sc["desc"] or "(none)")


def make_chunks(scenes):
    chunks, cur, size = [], [], 0
    for sc in scenes:
        r = render_scene(sc)
        if cur and (len(cur) >= MAX_SCENES or size + len(r) > MAX_CHARS):
            chunks.append(cur)
            cur, size = [], 0
        cur.append(sc)
        size += len(r)
    if cur:
        chunks.append(cur)
    return chunks


def user_message(chunk):
    return "Scenes:\n\n" + "\n".join(render_scene(sc) for sc in chunk)


# ---------- checking the model's answer ----------
def parse_response(text, expected):
    """Return (results, error, warnings). results is None when the answer is unusable."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        data = json.loads(t)
    except ValueError:
        i, j = t.find("["), t.rfind("]")
        if i < 0 or j <= i:
            return None, "no JSON array found", 0
        try:
            data = json.loads(t[i:j + 1])
        except ValueError:
            return None, "JSON could not be read", 0
    if not isinstance(data, list):
        return None, "answer is not a list", 0
    by_n = {}
    for o in data:
        if isinstance(o, dict) and "n" in o:
            try:
                by_n[int(o["n"])] = o
            except (TypeError, ValueError):
                pass
    missing = [n for n in expected if n not in by_n]
    if missing:
        return None, "missing scenes %s" % missing[:5], 0
    res, warn = {}, 0
    for n in expected:
        o = by_n[n]
        r = {}
        for k, (_, mx) in FIELDS.items():
            try:
                v = int(o.get(k, 0))
            except (TypeError, ValueError):
                return None, "bad value for %s in scene %d" % (k, n), 0
            if v < 0 or v > mx:
                v = max(0, min(mx, v))
                warn += 1
            r[k] = v
        conds = o.get("c", [])
        if not isinstance(conds, list):
            conds, warn = [], warn + 1
        r["c"] = sorted({c for c in conds if c in VOCAB})
        warn += len([c for c in conds if c not in VOCAB])
        res[n] = r
    return res, "", warn


# ---------- calling the model ----------
def make_client():
    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package is not installed. Run:  python -m pip install anthropic")
        raise SystemExit
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("No API key found in this Git Bash window. Set it first (see the instructions), then run again.")
        raise SystemExit
    return anthropic.Anthropic(max_retries=5)


def real_call(client, model, user, n_scenes):
    resp = client.messages.create(
        model=model, max_tokens=n_scenes * 60 + 300, system=SYSTEM,
        messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def fake_call(user, imdb_id, run):
    """Invented answers from simple keyword counts plus noise. For testing the pipeline only."""
    parts = re.split(r"\[SCENE (\d+)\]", user)
    first = parts[1] if len(parts) > 1 else "0"
    rnd = random.Random("%s|%s|%s" % (imdb_id, first, run))
    out = []
    for i in range(1, len(parts), 2):
        n, t = int(parts[i]), parts[i + 1].lower()

        def lvl(words, mx):
            v = min(mx, sum(t.count(w) for w in words))
            if rnd.random() < 0.15:
                v = max(0, min(mx, v + rnd.choice([-1, 1])))
            return v
        conds = [c for c, w in [("water", "water"), ("fire_explosions", "explo"), ("vehicles", "car "),
                                ("firearms", "gun"), ("animals", "dog")] if w in t and rnd.random() < 0.8]
        out.append({"n": n, "ex": lvl(["crowd", "people", "soldiers"], 3), "st": lvl(["fight", "punch", "chase", "crash"], 3),
                    "vfx": lvl(["explo", "creature", "ghost"], 3), "set": lvl(["palace", "ship", "lab", "court"], 3),
                    "cm": lvl(["uniform", "armor", "costume"], 3), "mu": lvl(["song", "band", "music"], 2), "c": conds})
    text = json.dumps(out)
    return text, len(SYSTEM + user) // 4, len(text) // 4


def do_job(job, ctx):
    ns = [sc["n"] for sc in job["chunk"]]
    user = user_message(job["chunk"])
    tin = tout = 0
    err = ""
    for attempt in range(1, 4):
        try:
            if ctx["fake"]:
                text, it, ot = fake_call(user, job["imdb_id"], job["run"])
            else:
                text, it, ot = real_call(ctx["client"], ctx["model"], user, len(ns))
        except Exception as e:
            name = type(e).__name__
            if name in ("AuthenticationError", "PermissionDeniedError"):
                raise Fatal("The API key was refused (%s). Check the key and try again." % name)
            if name == "NotFoundError":
                raise Fatal("The service does not know the model name '%s'. Ask which model to use." % ctx["model"])
            err = "api error: %s" % name
            time.sleep(min(30, 2 ** attempt))
            continue
        tin += it
        tout += ot
        res, err, warn = parse_response(text, ns)
        if res is not None:
            return {"ok": True, "imdb_id": job["imdb_id"], "first": ns[0], "last": ns[-1], "run": job["run"],
                    "model": ctx["model"], "prompt_version": PROMPT_VERSION, "tagger_version": TAGGER_VERSION,
                    "attempts": attempt, "in_tokens": tin, "out_tokens": tout, "warnings": warn,
                    "tags": {str(n): r for n, r in res.items()}}
    return {"ok": False, "imdb_id": job["imdb_id"], "first": ns[0], "last": ns[-1], "run": job["run"],
            "model": ctx["model"], "prompt_version": PROMPT_VERSION, "attempts": 3,
            "in_tokens": tin, "out_tokens": tout, "error": err}


# ---------- sets ----------
def load_set(name, tdir):
    if name == "all":
        return None
    p = os.path.join(tdir, "set_%s.txt" % name)
    if not os.path.exists(p):
        print("Set file not found:", p, "- run:  python scene_tagger.py plan")
        raise SystemExit
    with open(p, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def log_path(tdir, fake):
    return os.path.join(tdir, "runs_%s.jsonl" % ("FAKE" if fake else "real"))


def read_log(path, model):
    recs = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("prompt_version") == PROMPT_VERSION and r.get("model") == model:
                    recs.append(r)
    return recs


# ---------- commands ----------
def cmd_plan(args):
    here, raw, tdir = paths()
    out_dir = os.path.join(here, "outputs")
    feats, ok_use = {}, set()
    with open(os.path.join(out_dir, "p1_production_features_v1.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            feats[r["imdb_id"]] = r
    with open(os.path.join(out_dir, "p1_budget_table_v1.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["use_for_budget_estimator"] == "1":
                ok_use.add(r["imdb_id"])
    elig = sorted(i for i, r in feats.items()
                  if i in ok_use and r["parse_mode"] == "strict" and r["flag_low_confidence"] == "0")
    print("Scripts eligible for the pilot (usable budget, parsed normally):", len(elig))
    p200 = sorted(random.Random(42).sample(elig, min(200, len(elig))))
    mid = [i for i in p200 if 80 <= int(feats[i]["scene_count"]) <= 160]
    p5 = sorted(random.Random(7).sample(mid, min(5, len(mid))))
    for name, ids in (("pilot200", p200), ("pilot5", p5)):
        p = os.path.join(tdir, "set_%s.txt" % name)
        if os.path.exists(p):
            print("Keeping existing", p)
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(ids) + "\n")
            print("Wrote", p, "(%d scripts)" % len(ids))
    sets = {"pilot5": load_set("pilot5", tdir), "pilot200": load_set("pilot200", tdir), "all": None}
    stats = {k: {"scripts": 0, "scenes": 0, "chunks": 0, "chars": 0, "skipped": 0} for k in sets}
    print("Reading all scripts to count (about a minute)...")
    for sc in iter_scripts(raw):
        chunks = make_chunks(sc["scenes"])
        chars = sum(len(render_scene(s)) for s in sc["scenes"])
        for k, ids in sets.items():
            if ids is None or sc["imdb_id"] in ids:
                st = stats[k]
                if not sc["scenes"]:
                    st["skipped"] += 1
                    continue
                st["scripts"] += 1
                st["scenes"] += len(sc["scenes"])
                st["chunks"] += len(chunks)
                st["chars"] += chars
    print("\nEstimates (tokens are about characters / 4; the real count comes from the first test):")
    for k, st in stats.items():
        tin = (st["chars"] + st["chunks"] * len(SYSTEM)) // 4
        tout = st["scenes"] * OUT_TOK_PER_SCENE + st["chunks"] * 10
        print("\n[%s] scripts %d | scenes %d | calls per run %d | scripts with no scene text (skipped) %d" % (
            k, st["scripts"], st["scenes"], st["chunks"], st["skipped"]))
        for runs in (1, 3):
            line = "   %d run%s: %.1fM input tokens, %.1fM output tokens, %d calls" % (
                runs, "" if runs == 1 else "s", tin * runs / 1e6, tout * runs / 1e6, st["chunks"] * runs)
            if args.in_price is not None and args.out_price is not None:
                line += " | about $%.2f" % (tin * runs / 1e6 * args.in_price + tout * runs / 1e6 * args.out_price)
            print(line)
    if args.in_price is None:
        print("\nTo see dollars, add prices per MILLION tokens, e.g.:  python scene_tagger.py plan --in_price 1.00 --out_price 5.00")


def cmd_run(args):
    here, raw, tdir = paths()
    wanted = load_set(args.set, tdir)
    ctx = {"fake": args.fake, "model": "FAKE" if args.fake else args.model,
           "client": None if args.fake else make_client()}
    lp = log_path(tdir, args.fake)
    done = {(r["imdb_id"], r["first"], r["last"], r["run"]) for r in read_log(lp, ctx["model"]) if r.get("ok")}
    print("Model:", ctx["model"], "| prompt:", PROMPT_VERSION, "| runs per chunk:", args.runs,
          "| already done in log:", len(done))
    n_ok = n_fail = tin = tout = submitted = 0
    t0 = time.time()
    batch = []

    def flush(batch):
        nonlocal n_ok, n_fail, tin, tout
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=args.workers) as ex, open(lp, "a", encoding="utf-8") as out:
            futs = [ex.submit(do_job, j, ctx) for j in batch]
            try:
                for fu in as_completed(futs):
                    rec = fu.result()
                    rec["ts"] = int(time.time())
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    tin += rec["in_tokens"]
                    tout += rec["out_tokens"]
                    if rec["ok"]:
                        n_ok += 1
                    else:
                        n_fail += 1
                    if (n_ok + n_fail) % 25 == 0:
                        print("   %d calls done (%d failed) | %.2fM in / %.2fM out tokens | %.0f s" % (
                            n_ok + n_fail, n_fail, tin / 1e6, tout / 1e6, time.time() - t0))
            except Fatal as e:
                for fu in futs:
                    fu.cancel()
                print("\nSTOPPED:", e)
                print("Finished so far is saved. Fix the problem and run the same command again; it continues where it stopped.")
                raise SystemExit
        batch.clear()

    for sc in iter_scripts(raw, wanted):
        if not sc["scenes"]:
            print("Skipping (no scene text):", sc["movie_name"])
            continue
        for chunk in make_chunks(sc["scenes"]):
            for run in range(1, args.runs + 1):
                key = (sc["imdb_id"], chunk[0]["n"], chunk[-1]["n"], run)
                if key in done:
                    continue
                if args.max_chunks and submitted >= args.max_chunks:
                    break
                batch.append({"imdb_id": sc["imdb_id"], "chunk": chunk, "run": run})
                submitted += 1
        if len(batch) >= 200:
            flush(batch)
        if args.max_chunks and submitted >= args.max_chunks:
            break
    flush(batch)
    print("\nFinished. New successful calls: %d | failed calls: %d | tokens used this time: %.2fM in, %.2fM out" % (
        n_ok, n_fail, tin / 1e6, tout / 1e6))
    print("Log file:", lp)
    print("Next:  python scene_tagger.py report --set %s%s" % (args.set, " --fake" if args.fake else ""))


def cmd_report(args):
    here, raw, tdir = paths()
    model = "FAKE" if args.fake else args.model
    wanted = load_set(args.set, tdir)
    recs = read_log(log_path(tdir, args.fake), model)
    by_job = defaultdict(dict)       # (imdb, first, last) -> run -> tags
    warns = failed = tok_in = tok_out = 0
    for r in recs:
        if wanted is not None and r["imdb_id"] not in wanted:
            continue
        tok_in += r["in_tokens"]
        tok_out += r["out_tokens"]
        if r.get("ok"):
            by_job[(r["imdb_id"], r["first"], r["last"])][r["run"]] = r["tags"]
            warns += r.get("warnings", 0)
        else:
            failed += 1
    as_of = {}
    p = os.path.join(raw, "scene_parser_scripts.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                as_of[r["imdb_id"]] = r["as_of_date"]

    rows, expected_jobs, jobs_done = [], 0, 0
    levels = {k: Counter() for k in FIELDS}
    agree = {k: [0, 0] for k in FIELDS}
    spread = {k: 0.0 for k in FIELDS}
    cond_ct = Counter()
    scripts_done = scripts_total = n_multi = 0
    for sc in iter_scripts(raw, wanted):
        if not sc["scenes"]:
            continue
        scripts_total += 1
        full = True
        for chunk in make_chunks(sc["scenes"]):
            expected_jobs += 1
            runs = by_job.get((sc["imdb_id"], chunk[0]["n"], chunk[-1]["n"]), {})
            if runs:
                jobs_done += 1
            if len(runs) < args.runs:
                full = False
            if not runs:
                continue
            for s in chunk:
                per = [t[str(s["n"])] for t in runs.values() if str(s["n"]) in t]
                if not per:
                    continue
                row = {"imdb_id": sc["imdb_id"], "as_of_date": as_of.get(sc["imdb_id"], ""),
                       "movie_name": sc["movie_name"], "scene_no": s["n"], "heading": s["heading"][:80],
                       "int_ext": s["int_ext"], "location": s["location"], "day_night": s["day_night"],
                       "n_speaking_characters": s["n_speakers"], "dialogue_lines": s["dlines"],
                       "runs_used": len(per)}
                for k, (name, mx) in FIELDS.items():
                    vals = [x[k] for x in per]
                    mean = sum(vals) / len(vals)
                    row[name] = round(mean, 3)
                    levels[k][int(mean + 0.5)] += 1
                    if len(vals) > 1:
                        agree[k][1] += 1
                        agree[k][0] += 1 if len(set(vals)) == 1 else 0
                        spread[k] += max(vals) - min(vals)
                for c in VOCAB:
                    share = sum(1 for x in per if c in x["c"]) / len(per)
                    row["cond_" + c] = round(share, 3)
                    if share >= 0.5:
                        cond_ct[c] += 1
                if len(per) > 1:
                    n_multi += 1
                row["prompt_version"] = PROMPT_VERSION
                row["tagger_version"] = TAGGER_VERSION
                row["model"] = model
                row["_desc"] = s["desc"][:500]
                rows.append(row)
        if full:
            scripts_done += 1

    tag = "%s%s" % (args.set, "_FAKE" if args.fake else "")
    out_path = os.path.join(tdir, "scene_tags_%s.csv" % tag)
    hc = os.path.join(tdir, "handcheck_tags_%s.txt" % tag)
    if rows:
        cols = [c for c in rows[0].keys() if c != "_desc"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        with open(hc, "w", encoding="utf-8") as f:
            for r in random.Random(5).sample(rows, min(30, len(rows))):
                f.write("=" * 90 + "\n%s | scene %s | %s\n" % (r["movie_name"], r["scene_no"], r["heading"]))
                f.write("ACTION TEXT (first 500 characters): %s\n" % r["_desc"])
                f.write("MODEL SAYS: extras %s | stunts %s | vfx %s | set %s | costume/makeup %s | music %s | conditions: %s\n\n" % (
                    r["extras"], r["stunts"], r["vfx"], r["set_complexity"], r["costume_makeup"], r["music"],
                    ", ".join(c for c in VOCAB if r["cond_" + c] >= 0.5) or "none"))
    print("Set: %s | model: %s | prompt: %s" % (args.set, model, PROMPT_VERSION))
    print("Scripts with all %d runs complete: %d of %d" % (args.runs, scripts_done, scripts_total))
    print("Calls with at least one good answer: %d of %d | failed calls in log: %d | answers with fixed-up values: %d" % (
        jobs_done, expected_jobs, failed, warns))
    print("Tokens used (all attempts, this set): %.2fM in, %.2fM out" % (tok_in / 1e6, tok_out / 1e6))
    if args.in_price is not None and args.out_price is not None:
        print("Cost so far: about $%.2f" % (tok_in / 1e6 * args.in_price + tok_out / 1e6 * args.out_price))
    print("Scenes tagged:", len(rows))
    if not rows:
        return
    print("\nHow the scores are spread (share of scenes at each rounded level):")
    for k, (name, mx) in FIELDS.items():
        tot = sum(levels[k].values())
        print("   %-15s %s" % (name, "  ".join("L%d %4.1f%%" % (lv, 100.0 * levels[k][lv] / tot) for lv in range(mx + 1))))
    print("Special conditions (scenes where at least half the runs said yes):")
    print("   " + (", ".join("%s %d" % (c, n) for c, n in cond_ct.most_common()) if cond_ct else "none"))
    if n_multi:
        print("\nAgreement between runs (scenes with 2+ runs: %d). Exact-match rate, average gap:" % n_multi)
        for k, (name, _) in FIELDS.items():
            a, t = agree[k]
            print("   %-15s %5.1f%% identical | average gap %.2f" % (name, 100.0 * a / max(1, t), spread[k] / max(1, t)))
    print("\nSaved:", out_path)
    print("Saved:", hc, "(30 random scenes for you to judge)")


def main():
    sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "run", "report"])
    ap.add_argument("--set", default="pilot5")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--max_chunks", type=int, default=0)
    ap.add_argument("--in_price", type=float, default=None)
    ap.add_argument("--out_price", type=float, default=None)
    args = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
