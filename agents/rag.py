"""RAG codebase — retrieve tri thức KHÔNG train, KHÔNG tốn VRAM 3060.

Embedding chạy CPU bằng FastEmbed (ONNX) + vector store Chroma (nhúng, persistent) →
không cần thêm container GPU, đúng khuyến nghị cho 3060 (xem docs/multitask-strategy.md).

Dùng:
    python rag.py index                 # quét WORKDIR → chunk → embed → Chroma
    python rag.py search "hàm parse ..."# thử truy hồi
Trong agent: tool `search_codebase` (agents/tools.py) gọi rag.search().

Cấu hình qua env:
    AGENT_WORKDIR   thư mục codebase (mặc định .)
    RAG_DIR         nơi lưu Chroma (mặc định <WORKDIR>/.rag)
    RAG_EMBED_MODEL model embedding FastEmbed (mặc định BAAI/bge-small-en-v1.5;
                    đa ngữ/tiếng Việt: đặt BAAI/bge-m3 — nặng hơn)
"""
from __future__ import annotations
import argparse
import os
import pathlib

WORKDIR = pathlib.Path(os.environ.get("AGENT_WORKDIR", ".")).resolve()
RAG_DIR = os.environ.get("RAG_DIR", str(WORKDIR / ".rag"))
EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION = "codebase"

# Phần mở rộng file được index (code + tài liệu). Bỏ binary.
CODE_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".rs", ".c",
             ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".sh", ".yaml", ".yml",
             ".toml", ".json", ".md", ".txt", ".sql", ".vue", ".html", ".css"}
SKIP_DIRS = {".git", ".rag", "__pycache__", "node_modules", ".venv", "venv",
             "dist", "build", ".next", ".idea", "hf-cache"}


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Cắt văn bản thành các đoạn ~max_chars, gối nhau overlap ký tự. Thuần, test được."""
    text = text or ""
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]
    step = max(1, max_chars - overlap)
    out: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i:i + max_chars]
        if piece.strip():
            out.append(piece)
        if i + max_chars >= len(text):
            break
    return out


def iter_files(root: pathlib.Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = pathlib.Path(dirpath) / fn
            if p.suffix.lower() in CODE_EXTS:
                yield p


def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBED_MODEL)


def _collection():
    import chromadb
    client = chromadb.PersistentClient(path=RAG_DIR)
    # cosine → score = 1 - distance (dễ đọc)
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})


def _embed(texts: list[str]) -> list[list[float]]:
    return [list(map(float, v)) for v in _embedder().embed(texts)]


def index_workdir(max_chars: int = 1200, overlap: int = 150, batch: int = 256) -> int:
    """Index toàn bộ WORKDIR vào Chroma. Trả về số chunk đã ghi."""
    col = _collection()
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    total = 0

    def flush():
        nonlocal total
        if not docs:
            return
        embs = _embed(docs)
        col.upsert(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
        total += len(docs)
        ids.clear(); docs.clear(); metas.clear()

    for path in iter_files(WORKDIR):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(WORKDIR)).replace("\\", "/")
        for i, chunk in enumerate(chunk_text(text, max_chars, overlap)):
            ids.append(f"{rel}#{i}")
            docs.append(chunk)
            metas.append({"path": rel, "chunk": i})
            if len(docs) >= batch:
                flush()
    flush()
    return total


def search(query: str, top_k: int = 5) -> list[dict]:
    """Truy hồi top_k đoạn liên quan. Trả về [{path, snippet, score}]."""
    col = _collection()
    qemb = _embed([query])[0]
    res = col.query(query_embeddings=[qemb], n_results=top_k,
                    include=["documents", "metadatas", "distances"])
    out: list[dict] = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({
            "path": (meta or {}).get("path", "?"),
            "snippet": doc[:500],
            "score": round(1.0 - float(dist), 3),   # cosine: càng cao càng gần
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG codebase (index/search)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index", help="Index WORKDIR vào Chroma")
    pi.add_argument("--max-chars", type=int, default=1200)
    pi.add_argument("--overlap", type=int, default=150)
    ps = sub.add_parser("search", help="Thử truy hồi")
    ps.add_argument("query")
    ps.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    print(f"WORKDIR={WORKDIR}  RAG_DIR={RAG_DIR}  EMBED={EMBED_MODEL}")
    if args.cmd == "index":
        n = index_workdir(args.max_chars, args.overlap)
        print(f"✔ Đã index {n} chunk.")
    elif args.cmd == "search":
        for h in search(args.query, args.top_k):
            print(f"\n[{h['path']}] score={h['score']}\n{h['snippet']}")


if __name__ == "__main__":
    main()
