"""Test router fallback (agents/router.py). Cần openai + openai-agents → skip nếu chưa cài
(vd trên CI nhẹ). Chạy được trong venv của agents/."""
import pytest

pytest.importorskip("openai")
pytest.importorskip("agents")

import router
import skills
from openai import OpenAI


def test_registry_shape():
    assert skills.DEFAULT_SKILL in skills.SKILLS
    for s in skills.SKILLS.values():
        assert s.name and s.description and s.persona
        assert s.model in ("brain", "brain-pro") or s.model.startswith("brain-")


def test_router_fallback_on_dead_gateway():
    """Gateway chết → router KHÔNG raise, trả DEFAULT_SKILL an toàn."""
    dead = OpenAI(base_url="http://127.0.0.1:9/v1", api_key="x",
                  timeout=1, max_retries=0)
    name, why = router.choose_skill(dead, "sửa file rồi chạy test")
    assert name in skills.SKILLS
    assert name == skills.DEFAULT_SKILL
