#!/usr/bin/env python3
import argparse, json, math, os, re, sys, urllib.error, urllib.request
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

DOCS = ["product_overview.md", "PRODUCT_OVERVIEW_EN_int_A4_Web_v1_2.md", "PRODUCT OVERVIEW_EN_int_A4 Web_v1.2.md"]
REFUSAL = "The document does not contain enough information to answer this."
RRF_K = 60
COMMENT_RE = re.compile(r"<!--.*?-->")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TYPO_FIXES = [(r"\bofering\b", "offering"), (r"\bOfering\b", "Offering"), (r"\bofline\b", "offline"),
              (r"\bOfline\b", "Offline"), (r"\btrafic\b", "traffic"), (r"\bsensors an leaks\b", "sensors and leaks")]
STOPWORDS = set("a an the and or of to in on for with is are was were be been do does did can could i we you it its this "
                "that these those what which who whom how why when where at by from as into about over than then so if "
                "my our your me us their there any also should would will have has had not no please tell".split())
AGG_RE = re.compile(r"\b(all|which|list|every|each|compare|comparison|difference|differences|versus|vs|how many|what products|recommend|recommendation|suggest|best|fit|fits|suitable|options)\b", re.I)
SOURCES_RE = re.compile(r"(?im)[ \t]*\bsources[^\w\n]*:[^\w\n]*(.*)$")


def load_dotenv():
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        for name in (".env", "api.env"):
            f = base / name
            if f.exists():
                for line in f.read_text(encoding="utf-8-sig").splitlines():
                    line = line.strip()
                    if line.lower().startswith("export "):
                        line = line[7:].strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        v = v.strip().strip("'\"")
                        if v:
                            os.environ.setdefault(k.strip(), v)


def clean_line(s):
    s = IMAGE_RE.sub("", COMMENT_RE.sub("", s)).replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s).strip()
    m = re.match(r"^[•●▪‣◦·]\s*(.*)$", s) or re.match(r"^[-*]\s+(.*)$", s)
    if m:
        s = "- " + m.group(1).strip()
    for pat, rep in TYPO_FIXES:
        s = re.sub(pat, rep, s)
    return s


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


@dataclass
class Chunk:
    id: int
    kind: str
    name: str
    category: str
    text: str
    aliases: list[str] = field(default_factory=list)


def make_aliases(name):
    toks = norm(name).split()
    return [" ".join(toks)] + ([toks[-1]] if len(toks) > 1 and len(toks[-1]) >= 3 else [])


def parse_document(raw):
    entries = []
    for ln in raw.splitlines():
        s = clean_line(ln)
        if not s:
            continue
        h = HEADING_RE.match(s)
        entries.append((f"h{len(h.group(1))}", h.group(2).strip()) if h else ("bullet" if s.startswith("- ") else "text", s))
    for i, (t, x) in enumerate(entries):
        if (t == "text" and 0 < i < len(entries) - 1 and entries[i + 1][0] == "h3" and entries[i - 1][0] != "h3"
                and len(x.split()) <= 8 and not x.endswith(".")):
            entries[i] = ("h2", x)
    title, cats, cat, prod = "", [], None, None
    for t, x in entries:
        if t == "h1":
            title = title or x
        elif t == "h2" or (t == "h3" and cat is None):
            cat = {"name": x if t == "h2" else "General", "intro": [], "products": []}
            cats.append(cat)
            prod = None
        if t == "h3":
            prod = {"name": x, "lines": []}
            cat["products"].append(prod)
        elif t not in ("h1", "h2"):
            (prod["lines"] if prod is not None else cat["intro"] if cat is not None else []).append(x)
    return title, cats


def build_chunks(raw):
    title, cats = parse_document(raw)
    chunks = []
    for cat in cats:
        intro = "\n".join(cat["intro"])
        for p in cat["products"]:
            parts = [f"Category: {cat['name']}", f"Product: {p['name']}"]
            if intro:
                parts += ["Section-level information (applies to every product in this category):", intro]
            parts += ["Product details:", "\n".join(p["lines"])]
            chunks.append(Chunk(len(chunks), "product", p["name"], cat["name"], "\n".join(parts), make_aliases(p["name"])))
        listing = [f"- {p['name']}: {next((l for l in p['lines'] if not l.startswith('- ')), '')}".rstrip(": ") for p in cat["products"]]
        parts = [f"Category: {cat['name']}", "Section overview"] + ([intro] if intro else []) + ["Products in this section:", "\n".join(listing)]
        chunks.append(Chunk(len(chunks), "section", f"Section: {cat['name']}", cat["name"], "\n".join(parts)))
    return title, chunks


def tokenize(text):
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t
            for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b, self.n = k1, b, len(docs)
        self.tf, self.len = [Counter(d) for d in docs], [len(d) for d in docs]
        self.avg = sum(self.len) / max(self.n, 1)
        self.df = Counter(t for d in docs for t in set(d))

    def score(self, q):
        out = [0.0] * self.n
        for t in set(q):
            n = self.df.get(t, 0)
            if n:
                idf = math.log(1 + (self.n - n + 0.5) / (n + 0.5))
                for i, tf in enumerate(self.tf):
                    f = tf.get(t, 0)
                    if f:
                        out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return out


def normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def post_json(url, payload, headers=None, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:400]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}") from None


def ollama_conn():
    key = os.environ.get("OLLAMA_API_KEY")
    host = (os.environ.get("OLLAMA_HOST") or ("https://ollama.com" if key else "http://localhost:11434")).rstrip("/")
    host = host if host.startswith("http") else "http://" + host
    return host, ({"Authorization": f"Bearer {key}"} if key else {}), "ollama.com" in host


class STEmbedder:
    def __init__(self, model):
        from sentence_transformers import SentenceTransformer
        self.model, self.name = SentenceTransformer(model), f"sentence-transformers:{model}"
        self.qp = "Represent this sentence for searching relevant passages: " if "bge" in model.lower() else ""

    def embed_docs(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([self.qp + text], normalize_embeddings=True)[0].tolist()


class OllamaEmbedder:
    def __init__(self, model):
        self.host, self.headers, _ = ollama_conn()
        self.model, self.name = model, f"ollama:{model}"
        self.dp, self.qp = ("search_document: ", "search_query: ") if "nomic" in model else ("", "")
        self._embed(["ping"])

    def _embed(self, texts):
        return post_json(f"{self.host}/api/embed", {"model": self.model, "input": texts}, self.headers)["embeddings"]

    def embed_docs(self, texts):
        return self._embed([self.dp + t for t in texts])

    def embed_query(self, text):
        return self._embed([self.qp + text])[0]


def pick_embedder(choice):
    for c in [choice] if choice != "auto" else ["st", "ollama"]:
        try:
            if c == "none":
                return None
            return STEmbedder(os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")) if c == "st" else \
                OllamaEmbedder(os.environ.get("EMBED_MODEL", "nomic-embed-text"))
        except Exception:
            pass


class LLM:
    def __init__(self, model=None):
        self.host, self.headers, hosted = ollama_conn()
        self.hosted = hosted
        self.model = model or os.environ.get("LLM_MODEL") or ("gpt-oss:120b" if hosted else "llama3.1:8b")
        if hosted:
            self.model = re.sub(r"[:\-]cloud$", "", self.model)
        self.label = f"ollama:{self.model}"

    def chat(self, system, user):
        opts = {"temperature": 0} if self.hosted else {"temperature": 0, "num_ctx": 8192}
        r = post_json(f"{self.host}/api/chat", {"model": self.model, "stream": False, "options": opts, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}]}, self.headers)
        out = (r.get("message") or {}).get("content", "").strip()
        if not out:
            raise RuntimeError("Ollama returned an empty answer (reasoning models can exhaust max tokens).")
        return out


class Index:
    def __init__(self, chunks, embedder):
        self.chunks, self.embedder = chunks, embedder
        self.products = [c for c in chunks if c.kind == "product"]
        self.bm25 = BM25([tokenize(c.text) for c in chunks])
        self.vecs = [normalize(v) for v in embedder.embed_docs([c.text for c in chunks])] if embedder else None

    def detect_products(self, query):
        qn = norm(query)
        qt = qn.split()
        hits = []
        for c in self.products:
            hit = any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", qn) for a in c.aliases)
            if not hit:
                head, *rest = norm(c.name).split()
                hit = any(len(t) >= 6 and SequenceMatcher(None, t, head).ratio() >= (0.82 if rest else 0.85)
                          and qt[i + 1:i + 1 + len(rest)] == rest for i, t in enumerate(qt))
            if hit:
                hits.append(c)
        return hits

    def retrieve(self, query, pool=20):
        n, fused = len(self.chunks), Counter()
        bm = self.bm25.score(tokenize(query))
        dn = None
        if self.vecs:
            qv = normalize(self.embedder.embed_query(query))
            dn = [sum(x * y for x, y in zip(qv, v)) for v in self.vecs]
        for name, sc in (("bm25", bm), ("dense", dn)):
            if sc is not None:
                for r, i in enumerate(sorted(range(n), key=lambda i: -sc[i])[:pool]):
                    if not (name == "bm25" and sc[i] <= 0):
                        fused[i] += 1.0 / (RRF_K + r + 1)
        return [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])]

    def build_context(self, query, top_k=6):
        forced = self.detect_products(query)[:4]
        top_k = max(top_k, 10) if AGG_RE.search(query) else top_k
        ctx, seen = list(forced), {c.id for c in forced}
        for i in self.retrieve(query):
            if len(ctx) - len(forced) >= top_k:
                break
            if i not in seen:
                ctx.append(self.chunks[i])
                seen.add(i)
        return ctx, forced


def system_prompt(title):
    return f"""You are a product-knowledge assistant for a manufacturer's water leak detection equipment.
You answer ONLY from the CONTEXT excerpts of the document "{title}" that the user message provides.

Rules:
1. Use only facts stated in the context and stay close to the document's own wording. Never add descriptions, uses or
   qualities the text does not state (for example do not call a device "handheld", "portable" or suited to a particular
   use unless the text says so) and no qualifiers or intensifiers the text does not use (such as "general", "specialised",
   "less suitable"), and no hedging or speculation (may, might, could, likely, probably, suggests, appears). Every word
   that describes a product must come from the document. Never guess or infer specifications, prices, stock, availability, ordering or shipping
   status, or compatibility.
2. Use the exact sentence "{REFUSAL}" only when the context has nothing relevant to the question, and then never add
   anything else. If only part of the question is answerable, answer that part and add a separate sentence saying which
   part the document does not cover, without using the exact refusal sentence.
3. Attribute every feature to the correct product. Facts under "Product details" belong to that product only.
   "Section-level information" applies to all products of that category as written. Never move a feature from one
   product to another.
4. Yes/no questions ("does X use / have / support Y?"): begin with "Yes", "No" or "The document does not say".
   Answer "No" when the document describes a different technology or feature for X, or assigns Y to another product;
   then state what the document says about X and which product has Y. Use "does not say" only when the document is
   silent about both. This does not apply to questions about availability, ordering, price or dates; see rule 8.
5. For "which products..." questions, check every excerpt and list all that explicitly match, and only those.
6. Recommendations: go through EVERY excerpt and include each product whose text mentions any part of the need (pipe
   material, distance, installation conditions and so on), saying what the text states about it. Give the best fit first
   (the product matching the whole need), then all the others, each with its own caveat from the text, including
   "launching <date>; contact sales for pre-series units" when the text says a product is not yet launched. Include
   applicable section-level information using its wording (for example installation requirements).
7. Comparisons: cover the same aspects for each product, using only what the text states for each.
8. Availability, ordering, price and date questions ("can I order / buy X", "is X available", "how much"): never start
   with "Yes" or "No". Start with "The document does not say that ..." and then give the exact wording the document
   does contain (for example "launching Q2 2026; contact sales for pre-series units") without rewording it as
   "available" or "in stock". A future launch does not mean a product can be ordered now.
9. Plain text only: no markdown, no tables, no bold. Use short sentences or short "- " lists, no preamble. Use product
   names exactly as written in the document.
10. End with one final line: SOURCES: <names of the products / sections you used, exactly as in the context headers,
   separated by semicolons>, or SOURCES: none.

Style examples (fictional products, only to show the format):
Q: Does the ACME X1 use a laser sensor?
A: No. The document says the ACME X1 uses an ultrasonic sensor; laser sensing is described only for the ACME X2.
Q: Can I buy the ACME X3 now?
A: The document does not say that the ACME X3 can be ordered now. It only says the ACME X3 is "launching Q1 2030; contact sales for pre-series units"."""


def split_sources(answer, index, ctx):
    m = None
    for m in SOURCES_RE.finditer(answer):
        pass
    names = []
    if m:
        lookup = {}
        for c in index.chunks:
            lookup[c.name.lower()] = c.name
            lookup.setdefault(c.category.lower(), f"Section: {c.category}")
        for part in re.split(r"[;,]", m.group(1)):
            key = part.strip().strip("*. ").lower()
            ex = re.fullmatch(r"(?:excerpt\s*)?(\d+)", key)
            if ex and 0 < int(ex.group(1)) <= len(ctx):
                key = ctx[int(ex.group(1)) - 1].name.lower()
            key = re.sub(r"^section:\s*", "", key) if key not in lookup else key
            label = lookup.get(key) or lookup.get(part.strip().lower())
            if label and label not in names:
                names.append(label)
    return SOURCES_RE.sub("", answer).strip(), names


META = set("document says say said state states stated describe describes described mention mentions mentioned list lists listed "
           "contain contains information section level excerpt product products feature features recommend recommended "
           "recommendation yes enough answer only other others another each both either same different difference differences "
           "whereas versus vs while compared key summary therefore thus because however instead rather but they them one two "
           "more most less such just very provide provides provided offer offers highlight highlights highlighted refer refers".split())


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def wstem(w):
    for a, b in (("isation", "ization"), ("ising", "izing"), ("ised", "ized"), ("yse", "yze")):
        w = w.replace(a, b)
    while True:
        for suf in ("ations", "ation", "ions", "ion", "ings", "ing", "ors", "or", "ies", "ied", "ed", "es", "s", "ly"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[:-len(suf)]
                break
        else:
            return w


def supported(w, vocab, raw):
    if any(c.isdigit() for c in w):
        return w in raw
    s = wstem(w)
    return s in vocab or (len(s) >= 6 and any(abs(len(v) - len(s)) <= 2 and SequenceMatcher(None, s, v).ratio() >= 0.88 for v in vocab))


def unsupported(text, vocab, raw):
    return sorted({w for w in words(text) if len(w) > 1 and not supported(w, vocab, raw)})


def drop_unsupported(body, vocab, raw):
    out = []
    for line in body.splitlines():
        keep = [x for x in re.split(r"(?<=[.!?])\s+", line) if not unsupported(x, vocab, raw)]
        if keep or not line.strip():
            out.append(" ".join(keep))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() or REFUSAL


def answer_query(query, index, llm, title):
    ctx, forced = index.build_context(query)
    context = "\n\n".join(f"--- EXCERPT {i} ---\n{c.text}" for i, c in enumerate(ctx, 1))
    user = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer using only the context above and follow the rules."
    raw = set(words(f"{context} {query} {REFUSAL}"))
    vocab = {wstem(w) for w in raw | STOPWORDS | META}

    def ask(prompt):
        body, names = split_sources(llm.chat(system_prompt(title), prompt), index, ctx)
        return re.sub(r"\*\*|__", "", body), names

    body, names = ask(user)
    bad = unsupported(body, vocab, raw)
    if bad:
        try:
            body, names2 = ask(f"{user}\n\nDRAFT ANSWER:\n{body}\n\nThe draft uses words that appear neither in the document nor in "
                               f"the question: {', '.join(bad)}.\nRewrite the answer so that every statement uses only the document's "
                               "own wording, and delete every statement that relies on those words. Keep the final SOURCES line.")
            names = names2 or names
        except Exception:
            pass
        body = drop_unsupported(body, vocab, raw)
        names = [n for n in names if n.startswith("Section:") or n.lower() in body.lower()]
    if body.lower().startswith("the document does not contain"):
        names = []
    elif not names:
        names = [c.name for c in ctx if c.kind == "product" and c.name.lower() in body.lower()] or [c.name for c in forced]
    return body, names


def find_doc(arg):
    if arg:
        if Path(arg).exists():
            return Path(arg)
        sys.exit(f"Document not found: {arg}")
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for name in DOCS:
            if (base / name).exists():
                return base / name
        for p in sorted(base.glob("*.md")):
            if "product" in p.name.lower() and "overview" in p.name.lower():
                return p
    sys.exit("Could not find the markdown file. Put product_overview.md next to this script or pass --doc PATH.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc")
    ap.add_argument("--model")
    load_dotenv()
    args = ap.parse_args()
    doc = find_doc(args.doc)
    title, chunks = build_chunks(doc.read_text(encoding="utf-8-sig"))
    embedder = pick_embedder(os.environ.get("EMBED_BACKEND", "auto"))
    index, llm = Index(chunks, embedder), LLM(args.model)
    print(f"Loaded {len(chunks)} chunks | embeddings: {embedder.name if embedder else 'none (BM25 only)'} | llm: {llm.label}")
    print("Ask a question about the products. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        try:
            body, names = answer_query(q, index, llm, title)
        except Exception as e:
            print(f"Agent: [error] {e}\n")
            continue
        print(f"Agent: {body}")
        if names:
            print(f"[Sources: {'; '.join(names)}]")
        print()


if __name__ == "__main__":
    main()
