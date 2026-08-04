from __future__ import annotations

from datetime import date
from typing import Any


def _integer_component(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(text)
    except (TypeError, ValueError):
        return None


def crossref_date(message: dict[str, Any]) -> tuple[str | None, int | None]:
    """Return the most authoritative usable Crossref publication date.

    Crossref date-parts occasionally contain null or otherwise invalid month/day
    values. A valid year is retained while missing or invalid trailing components
    are discarded. A completely invalid date source is skipped so the next
    Crossref date field can be considered.
    """

    for key in ("published-print", "published-online", "published", "issued"):
        value = message.get(key)
        if not isinstance(value, dict):
            continue
        parts = value.get("date-parts")
        if not isinstance(parts, list) or not parts:
            continue
        first = parts[0]
        if not isinstance(first, (list, tuple)) or not first:
            continue

        year = _integer_component(first[0])
        if year is None or not 1 <= year <= 9999:
            continue

        month = _integer_component(first[1]) if len(first) > 1 else None
        if month is None or not 1 <= month <= 12:
            return f"{year:04d}", year

        day = _integer_component(first[2]) if len(first) > 2 else None
        if day is None or not 1 <= day <= 31:
            return f"{year:04d}-{month:02d}", year

        try:
            date(year, month, day)
        except ValueError:
            return f"{year:04d}-{month:02d}", year
        return f"{year:04d}-{month:02d}-{day:02d}", year

    return None, None
