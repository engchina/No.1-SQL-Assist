"""Local JSON cache for table/view/profile metadata."""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_METADATA_CACHE_VERSION = 1
_CACHE_DIR = Path("metadata_cache")
_METADATA_CACHE_FILE = _CACHE_DIR / "list_metadata.json"
_LEGACY_PROFILE_CACHE_FILE = Path("profiles") / "selectai.json"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def metadata_cache_path() -> Path:
    return _METADATA_CACHE_FILE


def _empty_cache() -> dict:
    return {
        "version": _METADATA_CACHE_VERSION,
        "updated_at": "",
        "table_views_updated_at": "",
        "profiles_updated_at": "",
        "tables": [],
        "views": [],
        "profiles": [],
    }


def _to_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_name(value) -> str:
    return _to_text(value).strip()


def _sort_by_name(entries: list, key: str) -> list:
    return sorted(entries, key=lambda item: _to_text(item.get(key)).upper())


def _normalize_table_entry(entry: dict) -> dict:
    rows = entry.get("rows", entry.get("Rows", ""))
    if rows is None:
        rows = ""
    elif not isinstance(rows, (str, int, float, bool)):
        rows = str(rows)
    return {
        "name": _normalize_name(entry.get("name", entry.get("Table Name", ""))),
        "rows": rows,
        "comments": _to_text(entry.get("comments", entry.get("Comments", ""))),
    }


def _normalize_view_entry(entry: dict) -> dict:
    return {
        "name": _normalize_name(entry.get("name", entry.get("View Name", ""))),
        "comments": _to_text(entry.get("comments", entry.get("Comments", ""))),
    }


def _normalize_profile_entry(entry: dict) -> dict:
    attrs = entry.get("attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    tables = entry.get("tables") or []
    views = entry.get("views") or []
    return {
        "profile": _normalize_name(entry.get("profile", entry.get("Profile Name", ""))),
        "category": _to_text(entry.get("category", entry.get("Category", ""))),
        "tables": [str(name) for name in tables if str(name).strip()],
        "views": [str(name) for name in views if str(name).strip()],
        "region": _to_text(entry.get("region", entry.get("Region", ""))),
        "model": _to_text(entry.get("model", entry.get("Model", ""))),
        "embedding_model": _to_text(
            entry.get("embedding_model", entry.get("Embedding Model", ""))
        ),
        "attributes": attrs,
    }


def _normalize_cache(cache: dict) -> dict:
    normalized = _empty_cache()
    if isinstance(cache, dict):
        normalized.update({k: v for k, v in cache.items() if k in normalized})
    normalized["version"] = _METADATA_CACHE_VERSION
    normalized["tables"] = [
        item
        for item in (_normalize_table_entry(entry) for entry in normalized["tables"])
        if item["name"]
    ]
    normalized["views"] = [
        item
        for item in (_normalize_view_entry(entry) for entry in normalized["views"])
        if item["name"]
    ]
    normalized["profiles"] = [
        item
        for item in (_normalize_profile_entry(entry) for entry in normalized["profiles"])
        if item["profile"]
    ]
    normalized["tables"] = _sort_by_name(normalized["tables"], "name")
    normalized["views"] = _sort_by_name(normalized["views"], "name")
    normalized["profiles"] = _sort_by_name(normalized["profiles"], "profile")
    return normalized


def load_metadata_cache() -> dict:
    if not _METADATA_CACHE_FILE.exists():
        cache = _empty_cache()
        cache["profiles"] = _load_legacy_profile_cache()
        return _normalize_cache(cache)
    try:
        with _METADATA_CACHE_FILE.open("r", encoding="utf-8") as f:
            return _normalize_cache(json.load(f) or {})
    except Exception as e:
        logger.error(f"load_metadata_cache error: {e}")
        return _empty_cache()


def save_metadata_cache(cache: dict) -> Path:
    normalized = _normalize_cache(cache)
    normalized["updated_at"] = _now_iso()
    _METADATA_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _METADATA_CACHE_FILE.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    tmp_path.replace(_METADATA_CACHE_FILE)
    return _METADATA_CACHE_FILE


def _load_legacy_profile_cache() -> list:
    """Read old profiles/selectai.json only for migration compatibility."""
    if not _LEGACY_PROFILE_CACHE_FILE.exists():
        return []
    try:
        with _LEGACY_PROFILE_CACHE_FILE.open("r", encoding="utf-8") as f:
            payload = json.load(f) or []
        profiles = []
        for entry in payload:
            normalized = _normalize_profile_entry(entry)
            if normalized.get("profile"):
                profiles.append(normalized)
        return profiles
    except Exception as e:
        logger.error(f"_load_legacy_profile_cache error: {e}")
        return []


def get_table_cache_entries() -> list:
    return load_metadata_cache().get("tables") or []


def get_view_cache_entries() -> list:
    return load_metadata_cache().get("views") or []


def get_profile_cache_entries() -> list:
    return load_metadata_cache().get("profiles") or []


def get_profile_cache_entry(display_or_name: str) -> dict:
    s = _normalize_name(display_or_name)
    if not s:
        return {}
    profiles = get_profile_cache_entries()
    for profile in profiles:
        if _normalize_name(profile.get("profile")) == s:
            return profile
    for profile in profiles:
        if _normalize_name(profile.get("category")) == s:
            return profile
    return {}


def replace_table_view_cache(tables: list, views: list) -> Path:
    cache = load_metadata_cache()
    cache["tables"] = tables or []
    cache["views"] = views or []
    cache["table_views_updated_at"] = _now_iso()
    return save_metadata_cache(cache)


def replace_profile_cache(profiles: list) -> Path:
    cache = load_metadata_cache()
    cache["profiles"] = profiles or []
    cache["profiles_updated_at"] = _now_iso()
    return save_metadata_cache(cache)


def _upsert_entry(entries: list, entry: dict, key: str) -> list:
    normalized_key = _normalize_name(entry.get(key)).upper()
    out = [
        current
        for current in entries
        if _normalize_name(current.get(key)).upper() != normalized_key
    ]
    out.append(entry)
    return out


def upsert_table_cache_entry(entry: dict) -> Path:
    cache = load_metadata_cache()
    normalized_entry = _normalize_table_entry(entry)
    if not normalized_entry["name"]:
        return save_metadata_cache(cache)
    cache["tables"] = _upsert_entry(cache.get("tables") or [], normalized_entry, "name")
    cache["table_views_updated_at"] = _now_iso()
    return save_metadata_cache(cache)


def upsert_view_cache_entry(entry: dict) -> Path:
    cache = load_metadata_cache()
    normalized_entry = _normalize_view_entry(entry)
    if not normalized_entry["name"]:
        return save_metadata_cache(cache)
    cache["views"] = _upsert_entry(cache.get("views") or [], normalized_entry, "name")
    cache["table_views_updated_at"] = _now_iso()
    return save_metadata_cache(cache)


def upsert_profile_cache_entry(entry: dict) -> Path:
    cache = load_metadata_cache()
    normalized_entry = _normalize_profile_entry(entry)
    if not normalized_entry["profile"]:
        return save_metadata_cache(cache)
    cache["profiles"] = _upsert_entry(
        cache.get("profiles") or [],
        normalized_entry,
        "profile",
    )
    cache["profiles_updated_at"] = _now_iso()
    return save_metadata_cache(cache)


def remove_table_cache_entry(name: str) -> Path:
    cache = load_metadata_cache()
    target = _normalize_name(name).upper()
    cache["tables"] = [
        entry
        for entry in cache.get("tables") or []
        if _normalize_name(entry.get("name")).upper() != target
    ]
    cache["table_views_updated_at"] = _now_iso()
    return save_metadata_cache(cache)


def remove_view_cache_entry(name: str) -> Path:
    cache = load_metadata_cache()
    target = _normalize_name(name).upper()
    cache["views"] = [
        entry
        for entry in cache.get("views") or []
        if _normalize_name(entry.get("name")).upper() != target
    ]
    cache["table_views_updated_at"] = _now_iso()
    return save_metadata_cache(cache)


def remove_profile_cache_entry(name: str) -> Path:
    cache = load_metadata_cache()
    target = _normalize_name(name).upper()
    cache["profiles"] = [
        entry
        for entry in cache.get("profiles") or []
        if _normalize_name(entry.get("profile")).upper() != target
    ]
    cache["profiles_updated_at"] = _now_iso()
    return save_metadata_cache(cache)
