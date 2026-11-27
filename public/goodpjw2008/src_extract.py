"""Extract programs that public notebooks publish in their *source* or attached datasets (for notebooks whose output
has no submission file, e.g. WRITE_SUBMISSION=False). Nothing from those notebooks is executed: we only parse CSV files,
base64+zlib JSON blobs and dict literals (json / ast.literal_eval), and every program is replayed by our referee."""
import ast, base64, csv, glob, io, json, lzma, os, re, zlib
from elf import replay

MAXB = 64 << 20
def _blob(b):
    """base64 of zlib- or xz-compressed JSON {id: program} or CSV text with id,program columns."""
    try: data = base64.b64decode(b)
    except Exception: return {}
    raw = None
    for dec in (lambda d: zlib.decompressobj().decompress(d, MAXB), lambda d: lzma.LZMADecompressor().decompress(d, MAXB)):
        try: raw = dec(data); break
        except Exception: continue
    if raw is None: return {}
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except Exception: pass
    try:
        rd = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if rd.fieldnames and "id" in rd.fieldnames and "program" in rd.fieldnames:
            return {r["id"]: json.loads(r["program"]) for r in rd}
    except Exception: pass
    try:   # compact text: "<id> a4,4,5 m5,5,6 ..." per line (a/s/m/d = + - * /)
        ops = {"a": "+", "s": "-", "m": "*", "d": "/"}; out = {}
        for line in raw.decode("utf-8", "ignore").split("\n"):
            parts = line.split()
            if not parts or not re.match(r"(?:scored|practice)_\d+$", parts[0]): continue
            out[parts[0]] = [[ops[t[0]], *map(int, t[1:].split(","))] for t in parts[1:] if t[0] in ops]
        return out
    except Exception: return {}

def _dicts(s):
    out = {}
    for m in re.finditer(r"\{\s*[\"'](?:scored|practice)_\d+[\"']\s*:", s):
        i = m.start(); depth = 0
        for j in range(i, min(len(s), i + (8 << 20))):
            c = s[j]
            if c in "{[": depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0: break
        try:
            d = ast.literal_eval(s[i:j + 1])
            if isinstance(d, dict): out.update(d)
        except Exception: pass
    return out

def candidates(folder):
    """{puzzle_id: [program, ...]} from every CSV / notebook under folder."""
    C = {}
    def add(k, p):
        if isinstance(k, str) and isinstance(p, list): C.setdefault(k, []).append(p)
    for f in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True):
        try:
            with open(f, newline="", encoding="utf-8-sig") as fh:
                rd = csv.DictReader(fh)
                if not rd.fieldnames or "id" not in rd.fieldnames or "program" not in rd.fieldnames: continue
                for r in rd:
                    try: add(r["id"], json.loads(r["program"]))
                    except Exception: pass
        except Exception: pass
    for f in glob.glob(os.path.join(folder, "**", "*.ipynb"), recursive=True):
        try: nb = json.load(open(f))
        except Exception: continue
        for c in nb.get("cells", []):
            if c.get("cell_type") != "code": continue
            s = "".join(c.get("source", []))
            for m in re.finditer(r"(?:'''|\"\"\")\s*([A-Za-z0-9+/=\s]{200,}?)\s*(?:'''|\"\"\")", s):
                for k, p in _blob(re.sub(r"\s", "", m.group(1))).items(): add(k, p)
            for m in re.finditer(r"[\"']([A-Za-z0-9+/=]{400,})[\"']", s):          # one-line blobs, e.g. NAME_XZ_B64 = "..."
                for k, p in _blob(m.group(1)).items(): add(k, p)
            for k, p in _dicts(s).items(): add(k, p)
    return C

def best_valid(P, C):
    out = {}
    for k, progs in C.items():
        if k not in P: continue
        for p in sorted(progs, key=len):
            if k in out and len(p) >= len(out[k]): break
            try: replay(P[k], [list(x) for x in p]); out[k] = [list(x) for x in p]; break
            except Exception: continue
    return out
