"""Test lõi eval matrix (training/eval_matrix.py) — checkers, aggregation, cổng cứng."""
import eval_matrix as em


def test_checkers():
    assert em.check_json_valid('{"a":1}')
    assert not em.check_json_valid('không phải json')
    assert em.check_tool_call_valid('<tool_call>\n{"name":"r","arguments":{}}\n</tool_call>')
    assert not em.check_tool_call_valid('quên gọi tool')
    assert not em.check_tool_call_valid('<tool_call>\n{"bad":1}\n</tool_call>')     # thiếu name/arguments
    assert not em.check_tool_call_valid('<tool_call>\n{bad json}\n</tool_call>')    # json lỗi
    assert em.check_contains('abc def', 'def') and not em.check_contains('abc', 'zzz')
    assert em.check_regex('ngày 2026-07-10', r'\d{4}-\d{2}-\d{2}')


def test_evaluate_keys_by_task_and_check():
    """Một task có 2 loại check phải TÁCH riêng, không trộn rate."""
    items = [
        {"task_id": "build", "messages": [], "check": "tool_call_valid"},
        {"task_id": "build", "messages": [], "check": "tool_call_valid"},
        {"task_id": "build", "messages": [], "check": "contains", "expect": "OK"},
    ]
    outs = iter(['<tool_call>\n{"name":"a","arguments":{}}\n</tool_call>',  # tool ok
                 'quên tool',                                              # tool fail
                 'kết quả OK'])                                            # contains ok
    m = em.evaluate(items, lambda msgs: next(outs), concurrency=1)
    assert ("build", "tool_call_valid") in m and ("build", "contains") in m
    assert m[("build", "tool_call_valid")]["rate"] == 0.5
    assert m[("build", "contains")]["rate"] == 1.0


def test_gate_fails_low_tool_call():
    m = {("build", "tool_call_valid"):
         {"task": "build", "check": "tool_call_valid", "n": 10, "pass": 5, "rate": 0.5}}
    ok, fails = em.gate(m)
    assert ok is False and "build" in fails[0]


def test_gate_passes_and_ignores_ungated_check():
    m = {
        ("build", "tool_call_valid"):
            {"task": "build", "check": "tool_call_valid", "n": 100, "pass": 96, "rate": 0.96},
        ("chat", "contains"):
            {"task": "chat", "check": "contains", "n": 10, "pass": 3, "rate": 0.3},
    }
    ok, fails = em.gate(m)
    assert ok is True and fails == []          # contains không có cổng cứng
