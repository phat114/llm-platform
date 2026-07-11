"""Test lõi RAG (agents/rag.py) — chunk_text thuần, không cần fastembed/chromadb.
`import rag` phải chạy được KHÔNG cần deps nặng (chúng được import lazy)."""
import rag


def test_chunk_text_short_and_empty():
    assert rag.chunk_text("abc") == ["abc"]
    assert rag.chunk_text("   ") == []
    assert rag.chunk_text("") == []


def test_chunk_text_overlap_covers_and_bounds():
    text = "".join(str(i % 10) for i in range(3000))
    chunks = rag.chunk_text(text, max_chars=1000, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0].startswith(text[:10])   # bắt đầu từ đầu chuỗi
    assert text[-1] in chunks[-1]            # phủ tới cuối chuỗi


def test_exts_and_skipdirs():
    assert ".py" in rag.CODE_EXTS and ".md" in rag.CODE_EXTS
    assert ".git" in rag.SKIP_DIRS and "node_modules" in rag.SKIP_DIRS
