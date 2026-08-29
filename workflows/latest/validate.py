"""Validate the additive canonical-workflow index without running experiments."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("manifest.json")
VALID_ROLES = {"canonical", "supporting", "validation", "historical"}


def main() -> int:
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors: list[str] = []
    seen_ids: set[str] = set()
    entry_count = 0

    for authority in document.get("authorities", []):
        if not (ROOT / authority).is_file():
            errors.append(f"missing authority: {authority}")

    for stage in document.get("stages", []):
        stage_id = stage.get("id", "<missing>")
        runbook = stage.get("runbook", "")
        if not runbook or not (ROOT / runbook).is_file():
            errors.append(f"{stage_id}: missing runbook: {runbook}")

        for entry in stage.get("entrypoints", []):
            entry_count += 1
            entry_id = entry.get("id", "")
            qualified_id = f"{stage_id}.{entry_id}"
            if qualified_id in seen_ids:
                errors.append(f"duplicate entry id: {qualified_id}")
            seen_ids.add(qualified_id)

            role = entry.get("role")
            if role not in VALID_ROLES:
                errors.append(f"{qualified_id}: invalid role: {role}")

            path = entry.get("path", "")
            if not path or not (ROOT / path).is_file():
                errors.append(f"{qualified_id}: missing source: {path}")

    if errors:
        print("Canonical workflow validation FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        f"Canonical workflow validation passed: "
        f"{len(document['stages'])} stages, {entry_count} entry points."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

