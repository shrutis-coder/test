"""
Convert raw interview transcripts into one clean JSONL file.

Reads:
    raw/interviews/filmcrowd/*_transcript.txt   (Google Meet transcript tab, downloaded as .txt)
    raw/interviews/youtube/*.txt                (YouTube transcripts)
    raw/interviews/artists.csv                  (optional: who each file is, see bottom of this file)

Writes:
    clean/interviews/transcripts.jsonl          one line per transcript
    clean/interviews/dropped_log.csv            every line or file we skipped, and why

Run from the repo root:
    python src/interviews/convert_interviews.py
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

RAW_DIR = Path("raw/interviews")
OUT_DIR = Path("clean/interviews")

# People on the FilmCrowd side of the call. Their turns are kept (for context)
# but marked "interviewer" so the claim extractor only pulls claims from artists.
INTERVIEWERS = {"Shruti Srinivasan", "Sarah Agrawal"}

TIMESTAMP_LINE = re.compile(r"^\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?:?$")       # 00:00:02 or [1:23]
LEADING_TIMESTAMP = re.compile(r"^\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]:?\s*(.*)$")  # [0:11]: some text

# Caption files have no speaker names, so instead of one giant turn we group
# lines into ~60-second segments. Each keeps its start time so quotes stay checkable.
SEGMENT_SECONDS = 60
SPEAKER_LINE = re.compile(r"^([A-Z][^:\n]{0,40}):\s+(.*)$")                 # Name: text

# Google Meet boilerplate that is not part of the conversation.
BOILERPLATE = [
    re.compile(r"^Meeting .* - Transcript$"),
    re.compile(r"^Meeting ended after", re.I),
    re.compile(r"^Transcription ended after", re.I),
    re.compile(r"^This editable transcript was computer generated", re.I),
]

dropped = []  # rows for dropped_log.csv


def log_drop(file, line_no, reason, text):
    dropped.append({"file": file, "line": line_no, "reason": reason, "text": text[:200]})


def to_seconds(h_or_m, m_or_s, s=None):
    if s is None:  # mm:ss
        return int(h_or_m) * 60 + int(m_or_s)
    return int(h_or_m) * 3600 + int(m_or_s) * 60 + int(s)


def read_lines(path):
    # utf-8-sig strips the invisible BOM that Google Docs puts at the start of .txt exports
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def load_artists():
    """Optional raw/interviews/artists.csv with columns: file, artist, artist_role, artist_speaker_label"""
    path = RAW_DIR / "artists.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as f:
        return {row["file"].strip(): row for row in csv.DictReader(f)}


def parse_turns(path, known_interviewers, detect_speakers=True):
    """Turn a transcript into a list of speaker turns, merging consecutive lines by the same speaker."""
    turns = []
    current_sec = None
    date = None

    for i, raw in enumerate(read_lines(path), start=1):
        line = raw.strip()
        if not line:
            continue

        if any(p.search(line) for p in BOILERPLATE):
            log_drop(path.name, i, "boilerplate", line)
            continue

        if date is None and re.match(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}$", line):  # "Jun 12, 2026"
            date = line
            continue

        ts = TIMESTAMP_LINE.match(line)
        if ts:
            current_sec = to_seconds(*ts.groups())
            continue

        lead = LEADING_TIMESTAMP.match(line)
        if lead:
            current_sec = to_seconds(*lead.groups()[:3])
            line = lead.group(4)

        # YouTube captions never have speaker names, so a sentence like
        # "You've heard it all before: ..." must not be mistaken for one.
        sp = SPEAKER_LINE.match(line) if detect_speakers else None
        if sp:
            speaker, text = sp.group(1).strip(), sp.group(2).strip()
        elif turns:
            speaker, text = turns[-1]["speaker"], line  # continuation of the previous speaker
        else:
            speaker, text = None, line

        if not text:
            continue

        same_speaker = turns and turns[-1]["speaker"] == speaker
        segment_full = (
            speaker is None and turns and current_sec is not None
            and turns[-1]["start_sec"] is not None
            and current_sec - turns[-1]["start_sec"] >= SEGMENT_SECONDS
        )
        if same_speaker and not segment_full:
            turns[-1]["text"] += " " + text
        else:
            turns.append({"speaker": speaker, "start_sec": current_sec, "text": text})

    for n, t in enumerate(turns):
        t["turn_id"] = n
        if t["speaker"] is None:
            t["role"] = "unknown"
        elif t["speaker"] in known_interviewers:
            t["role"] = "interviewer"
        else:
            t["role"] = "artist"
    return turns, date


def guess_artist(turns):
    """The non-interviewer who talks the most is most likely the artist."""
    words = Counter()
    for t in turns:
        if t["role"] == "artist":
            words[t["speaker"]] += len(t["text"].split())
    return words.most_common(1)[0][0] if words else None


def build_record(path, source_type, artists, interviewers):
    turns, date = parse_turns(path, interviewers, detect_speakers=(source_type != "youtube"))
    meta = artists.get(path.name, {})

    # If artists.csv names the artist's speaker label, anyone else who isn't a
    # known interviewer is marked "other" (e.g. a second host on a YouTube show).
    label = (meta.get("artist_speaker_label") or "").strip()
    if label:
        for t in turns:
            if t["role"] == "artist" and t["speaker"] != label:
                t["role"] = "other"

    has_speakers = any(t["speaker"] for t in turns)
    return {
        "transcript_id": f"{'fc' if source_type == 'filmcrowd_interview' else 'yt'}_{path.stem}",
        "source_type": source_type,
        "file": str(path),
        "artist": (meta.get("artist") or "").strip() or guess_artist(turns),
        "artist_role": (meta.get("artist_role") or "").strip() or None,
        "date": date,
        "transcript_quality": "google_meet_auto" if source_type == "filmcrowd_interview" else "youtube_caption",
        "has_speaker_labels": has_speakers,
        "n_turns": len(turns),
        "n_words": sum(len(t["text"].split()) for t in turns),
        "turns": turns,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artists = load_artists()
    records = []

    for path in sorted((RAW_DIR / "filmcrowd").glob("*")):
        if path.suffix.lower() != ".txt":
            log_drop(path.name, None, "not a .txt file (convert or transcribe first)", "")
        elif "_notes" in path.stem:
            log_drop(path.name, None, "AI meeting notes, not the artist's own words", "")
        else:
            records.append(build_record(path, "filmcrowd_interview", artists, INTERVIEWERS))

    for path in sorted((RAW_DIR / "youtube").glob("*")):
        if path.suffix.lower() != ".txt":
            log_drop(path.name, None, "not a .txt file", "")
        else:
            records.append(build_record(path, "youtube", artists, set()))

    with (OUT_DIR / "transcripts.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with (OUT_DIR / "dropped_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "line", "reason", "text"])
        w.writeheader()
        w.writerows(dropped)

    # Quick summary so you can eyeball it
    print(f"Wrote {len(records)} transcripts to {OUT_DIR / 'transcripts.jsonl'}")
    print(f"Logged {len(dropped)} dropped items to {OUT_DIR / 'dropped_log.csv'}\n")
    print(f"{'transcript_id':45} {'artist':22} {'turns':>5} {'words':>7}  speakers?")
    for r in records:
        flag = "yes" if r["has_speaker_labels"] else "NO"
        print(f"{r['transcript_id'][:45]:45} {str(r['artist'])[:22]:22} {r['n_turns']:>5} {r['n_words']:>7}  {flag}")
    missing = [Path(r["file"]).name for r in records if not r["artist"]]
    if missing:
        template = RAW_DIR / "artists_template.csv"
        with template.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "artist", "artist_role", "artist_speaker_label"])
            for name in missing:
                w.writerow([name, "", "", ""])
        print(f"\n{len(missing)} transcripts have no artist name.")
        print(f"Wrote {template} with those file names filled in. Fill in the artist and role")
        print("columns, then rename it to artists.csv (or paste its rows into artists.csv).")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# raw/interviews/artists.csv (optional, but you'll want it for YouTube files)
#
# file,artist,artist_role,artist_speaker_label
# barrie_2_transcript.txt,Barrie Osborne,producer,Barrie Osborne
# di morrissey_transcript.txt,Di Morrissey,writer,DI
# Avatar interview 1.txt,James Cameron,director,
#
# "file" must match the file name exactly. Leave artist_speaker_label blank if
# the transcript has no speaker names.
# ---------------------------------------------------------------------------
