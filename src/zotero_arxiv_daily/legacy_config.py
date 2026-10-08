"""Translate the fork's existing Actions environment without exposing values."""
from collections.abc import Mapping
from pathlib import Path
import os

from omegaconf import DictConfig, OmegaConf


class LegacyConfigError(ValueError):
    """A migration decision is needed before production access."""


def build_legacy_config(env: Mapping[str, str], base: DictConfig) -> DictConfig:
    config = OmegaConf.create(OmegaConf.to_container(base, resolve=False))

    def value(key: str, default=None):
        return env.get(key) or default

    def required(key: str):
        result = value(key)
        if result is None:
            raise LegacyConfigError(f"Missing existing setting: {key}")
        return result

    def integer(key: str, default=None):
        try:
            return int(value(key, default))
        except (ValueError, TypeError):
            raise LegacyConfigError(f"Invalid integer setting: {key}") from None

    # The old parser accepted only true/1, and defaulted to local LLM.
    # Do not activate a paid API for users who selected local inference.
    if value("USE_LLM_API", "false").lower() not in ("true", "1"):
        raise LegacyConfigError(
            "USE_LLM_API selects the retired local LLM path. Keep the draft unmerged "
            "until local LLM support or an explicit API migration is approved."
        )

    config.zotero.user_id = required("ZOTERO_ID")
    config.zotero.api_key = required("ZOTERO_KEY")
    config.zotero.include_path = None
    config.zotero.ignore_path = None
    config.zotero.legacy_ignore = value("ZOTERO_IGNORE")
    config.email.sender = required("SENDER")
    config.email.receiver = required("RECEIVER")
    config.email.sender_password = required("SENDER_PASSWORD")
    config.email.smtp_server = required("SMTP_SERVER")
    config.email.smtp_port = integer("SMTP_PORT")
    config.llm.api.key = required("OPENAI_API_KEY")
    config.llm.api.base_url = value("OPENAI_API_BASE", "https://api.openai.com/v1")
    config.llm.api_mode = "chat_completion"
    config.llm.generation_kwargs = {"model": value("MODEL_NAME", "gpt-4o"), "temperature": 0}
    config.llm.language = value("LANGUAGE", "English")
    config.reranker.local.model = "avsolatorio/GIST-small-Embedding-v0"
    config.reranker.local.encode_kwargs = {}
    config.executor.reranker = "local"
    config.executor.source = ["arxiv"]
    config.executor.debug = False
    config.executor.send_empty = value("SEND_EMPTY", "false").lower() in ("true", "1")
    config.executor.max_paper_num = integer("MAX_PAPER_NUM", "100")
    if config.executor.max_paper_num < -1:
        raise LegacyConfigError("MAX_PAPER_NUM must be -1 or nonnegative")
    # Keep the original RSS query verbatim: no guessed category conversion.
    config.source.arxiv.category = [required("ARXIV_QUERY")]
    config.source.arxiv.include_cross_list = False
    return config


def main():
    # Avoid Hydra output directories/config snapshots containing resolved secrets.
    from loguru import logger

    base = OmegaConf.load(Path(__file__).resolve().parents[2] / "config/base.yaml")
    try:
        config = build_legacy_config(os.environ, base)
    except LegacyConfigError as exc:
        logger.error(str(exc))
        raise SystemExit(2) from None
    from .executor import Executor
    Executor(config).run()


if __name__ == "__main__":
    main()
