#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd -- "${script_dir}/.." && pwd)

usage() {
    cat <<'EOF'
Usage:
  scripts/migrate_selectai_categories.sh [LEGACY_JSON] [METADATA_JSON]

Defaults:
  LEGACY_JSON   profiles/selectai.json
  METADATA_JSON metadata_cache/list_metadata.json

Copies non-empty category values from the legacy SelectAI JSON into the
metadata cache by matching profile names. The destination is backed up before
an atomic write. All destination fields other than category are preserved.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if (( $# > 2 )); then
    usage >&2
    exit 2
fi

legacy_json=${1:-"${repository_root}/profiles/selectai.json"}
metadata_json=${2:-"${repository_root}/metadata_cache/list_metadata.json"}
python_bin=$(command -v python3 || true)

if [[ -z "${python_bin}" ]]; then
    echo "错误: 找不到 python3。" >&2
    exit 1
fi

"${python_bin}" - "${legacy_json}" "${metadata_json}" <<'PY'
import json
import os
import shutil
import stat
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def fail(message):
    print(f"错误: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path, label):
    if not path.is_file():
        fail(f"{label}不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except json.JSONDecodeError as exc:
        fail(f"{label}不是有效的 JSON: {path} ({exc})")
    except OSError as exc:
        fail(f"无法读取{label}: {path} ({exc})")


def normalized_profile_name(value):
    return str(value or "").strip().casefold()


legacy_path = Path(sys.argv[1]).expanduser()
metadata_path = Path(sys.argv[2]).expanduser()

legacy_payload = load_json(legacy_path, "旧 SelectAI JSON")
metadata_payload = load_json(metadata_path, "metadata cache JSON")

if not isinstance(legacy_payload, list):
    fail("旧 SelectAI JSON 的顶层必须是数组。")
if not isinstance(metadata_payload, dict):
    fail("metadata cache JSON 的顶层必须是对象。")

metadata_profiles = metadata_payload.get("profiles")
if not isinstance(metadata_profiles, list):
    fail("metadata cache JSON 的 profiles 必须是数组。")

legacy_categories = {}
legacy_display_names = {}
skipped_empty = 0

for index, entry in enumerate(legacy_payload):
    if not isinstance(entry, dict):
        fail(f"旧 SelectAI JSON 的第 {index + 1} 项必须是对象。")

    profile_name = str(entry.get("profile") or "").strip()
    category = str(entry.get("category") or "").strip()
    if not profile_name or not category:
        skipped_empty += 1
        continue

    profile_key = normalized_profile_name(profile_name)
    previous_category = legacy_categories.get(profile_key)
    if previous_category is not None and previous_category != category:
        fail(
            "旧 SelectAI JSON 中存在 category 不一致的重复 profile: "
            f"{profile_name}"
        )

    legacy_categories[profile_key] = category
    legacy_display_names[profile_key] = profile_name

matched_keys = set()
updated = 0
unchanged = 0

for index, entry in enumerate(metadata_profiles):
    if not isinstance(entry, dict):
        fail(f"metadata cache JSON 的 profiles 第 {index + 1} 项必须是对象。")

    profile_key = normalized_profile_name(entry.get("profile"))
    if not profile_key or profile_key not in legacy_categories:
        continue

    matched_keys.add(profile_key)
    legacy_category = legacy_categories[profile_key]
    if str(entry.get("category") or "") == legacy_category:
        unchanged += 1
        continue

    entry["category"] = legacy_category
    updated += 1

unmatched_keys = sorted(set(legacy_categories) - matched_keys)

print(f"匹配到的 profile: {len(matched_keys)}")
print(f"更新的 category: {updated}")
print(f"无需更新: {unchanged}")
print(f"跳过空 profile/category: {skipped_empty}")

if unmatched_keys:
    unmatched_names = ", ".join(legacy_display_names[key] for key in unmatched_keys)
    print(f"目标缓存中未找到: {unmatched_names}")

if updated == 0:
    print("没有需要写入的变更。")
    raise SystemExit(0)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
backup_path = metadata_path.with_name(f"{metadata_path.name}.bak.{timestamp}")
backup_index = 1
while backup_path.exists():
    backup_path = metadata_path.with_name(
        f"{metadata_path.name}.bak.{timestamp}.{backup_index}"
    )
    backup_index += 1

temporary_path = None
try:
    shutil.copy2(metadata_path, backup_path)
    original_mode = stat.S_IMODE(metadata_path.stat().st_mode)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=metadata_path.parent,
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(metadata_payload, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())

    os.chmod(temporary_path, original_mode)
    os.replace(temporary_path, metadata_path)
except OSError as exc:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
    fail(f"写入 metadata cache 失败: {exc}")

print(f"备份文件: {backup_path}")
print(f"迁移完成: {metadata_path}")
PY
