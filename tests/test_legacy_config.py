"""Migration regression tests use synthetic values only."""
from datetime import datetime
from types import SimpleNamespace
import runpy
import socket

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.legacy_config import build_legacy_config, LegacyConfigError


@pytest.fixture
def legacy_env():
    return {
        "USE_LLM_API": "true", "ZOTERO_ID": "000000", "ZOTERO_KEY": "fake-key",
        "SENDER": "test@example.com", "RECEIVER": "test@example.com",
        "SENDER_PASSWORD": "fake-password", "SMTP_SERVER": "localhost",
        "SMTP_PORT": "1025", "OPENAI_API_KEY": "fake-api-key",
        "ARXIV_QUERY": "cs.AI+cs.LG", "ZOTERO_IGNORE": "archive/**\n!archive/keep",
        "SEND_EMPTY": "1", "MAX_PAPER_NUM": "-1", "LANGUAGE": "Chinese",
        "MODEL_NAME": "existing-model", "OPENAI_API_BASE": "http://localhost:30000/v1",
    }


def legacy_config(env):
    return build_legacy_config(env, OmegaConf.load("config/base.yaml"))


def test_preserves_existing_settings(legacy_env):
    cfg = legacy_config(legacy_env)
    assert cfg.zotero.user_id == legacy_env["ZOTERO_ID"]
    assert cfg.zotero.api_key == legacy_env["ZOTERO_KEY"]
    assert cfg.zotero.legacy_ignore == legacy_env["ZOTERO_IGNORE"]
    assert cfg.zotero.ignore_path is None
    assert cfg.email.smtp_port == 1025
    for name in ["sender", "receiver", "sender_password", "smtp_server"]:
        assert cfg.email[name] == legacy_env[name.upper()]
    assert cfg.llm.api.key == legacy_env["OPENAI_API_KEY"]
    assert cfg.llm.api.base_url == legacy_env["OPENAI_API_BASE"]
    assert cfg.llm.language == "Chinese"
    assert cfg.llm.generation_kwargs == {"model": "existing-model", "temperature": 0}
    assert cfg.reranker.local.model == "avsolatorio/GIST-small-Embedding-v0"
    assert cfg.reranker.local.encode_kwargs == {}
    assert cfg.source.arxiv.category == ["cs.AI+cs.LG"]
    assert not cfg.source.arxiv.include_cross_list
    assert cfg.executor.source == ["arxiv"]
    assert cfg.executor.send_empty
    assert cfg.executor.max_paper_num == -1
    assert not OmegaConf.missing_keys(cfg)


def test_empty_values_keep_legacy_defaults(legacy_env):
    for name in ["MODEL_NAME", "OPENAI_API_BASE", "LANGUAGE", "SEND_EMPTY", "MAX_PAPER_NUM"]:
        legacy_env[name] = ""
    cfg = legacy_config(legacy_env)
    assert cfg.llm.generation_kwargs.model == "gpt-4o"
    assert cfg.llm.api.base_url == "https://api.openai.com/v1"
    assert cfg.llm.language == "English"
    assert not cfg.executor.send_empty
    assert cfg.executor.max_paper_num == 100


@pytest.mark.parametrize("mode", ["", "false", "0", "unexpected"])
def test_local_llm_requires_explicit_migration(legacy_env, mode):
    legacy_env["USE_LLM_API"] = mode
    with pytest.raises(LegacyConfigError, match="retired local LLM"):
        legacy_config(legacy_env)


@pytest.mark.parametrize("name", ["SMTP_PORT", "MAX_PAPER_NUM"])
def test_errors_do_not_echo_values(legacy_env, name):
    legacy_env[name] = "private-sentinel-value"
    with pytest.raises(LegacyConfigError) as exc:
        legacy_config(legacy_env)
    assert "private-sentinel-value" not in str(exc.value)


def test_missing_required_settings_fail_before_executor(legacy_env):
    del legacy_env["ARXIV_QUERY"]
    with pytest.raises(LegacyConfigError, match="Missing existing setting: ARXIV_QUERY"):
        legacy_config(legacy_env)


def test_custom_config_is_not_implicitly_activated(legacy_env):
    legacy_env["CUSTOM_CONFIG"] = "email:\n  smtp_server: unexpected.example"
    assert legacy_config(legacy_env).email.smtp_server == "localhost"


def test_legacy_gitignore_preserves_negation_and_any_collection(legacy_env):
    from zotero_arxiv_daily.executor import Executor
    from zotero_arxiv_daily.protocol import CorpusPaper
    cfg = legacy_config(legacy_env)
    executor = Executor.__new__(Executor)
    executor.config = cfg
    executor.include_path_patterns = None
    executor.ignore_path_patterns = None
    paths = [["archive/topic"], ["archive/keep"], ["active", "archive/topic"], []]
    corpus = [CorpusPaper(str(i), "", datetime(2026, 1, 1), p) for i, p in enumerate(paths)]
    assert [p.title for p in executor.filter_corpus(corpus)] == ["1", "3"]


def test_unlimited_papers_keeps_last_result(config, monkeypatch):
    from zotero_arxiv_daily.executor import Executor
    from tests.canned_responses import make_sample_paper
    from zotero_arxiv_daily.protocol import CorpusPaper
    executor = Executor.__new__(Executor)
    executor.config = config
    config.executor.max_paper_num = -1
    corpus = [CorpusPaper("Corpus", "Abstract", datetime(2026, 1, 1), [])]
    papers = [make_sample_paper(title=str(i)) for i in range(3)]
    for paper in papers:
        monkeypatch.setattr(paper, "generate_tldr", lambda *a: None)
        monkeypatch.setattr(paper, "generate_affiliations", lambda *a: None)
    executor.openai_client = None
    monkeypatch.setattr(executor, "fetch_zotero_corpus", lambda: corpus)
    monkeypatch.setattr(executor, "filter_corpus", lambda c: c)
    executor.retrievers = {"arxiv": SimpleNamespace(retrieve_papers=lambda: papers)}
    executor.reranker = SimpleNamespace(rerank=lambda p, c: p)
    rendered = []
    monkeypatch.setattr("zotero_arxiv_daily.executor.render_email", lambda p: rendered.extend(p) or "html")
    monkeypatch.setattr("zotero_arxiv_daily.executor.send_email", lambda *a: None)
    executor.run()
    assert len(rendered) == 3


@pytest.mark.parametrize("repository,ref,expected", [
    ("", "", True), ("MarkMrLi/zotero-arxiv-daily", "main", True),
    ("MarkMrLi/zotero-arxiv-daily", "refs/heads/main", True),
    ("TideDra/zotero-arxiv-daily", "main", False),
    ("", "old-commit", False),
])
def test_checkout_guard(repository, ref, expected):
    compatible = runpy.run_path("scripts/check_checkout.py")["compatible_checkout"]
    assert compatible({"CHECKOUT_REPOSITORY": repository, "CHECKOUT_REF": ref,
                       "GITHUB_REPOSITORY": "MarkMrLi/zotero-arxiv-daily"}) is expected


def test_offline_guard_blocks_real_connections():
    import os
    if os.environ.get("ZAD_OFFLINE_TESTS") != "1":
        pytest.skip("Offline guard not enabled")
    with socket.socket() as connection:
        with pytest.raises(AssertionError, match="Real network"):
            connection.connect(("127.0.0.1", 1))
