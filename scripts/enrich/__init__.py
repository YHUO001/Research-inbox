"""Metadata enrichment package."""

from scripts.enrich.crossref_dates import crossref_date as _safe_crossref_date
from scripts.enrich import metadata as _metadata

# Crossref sometimes emits null/invalid trailing date-parts. Patch the provider
# module at package import so all existing enrichment entry points use the
# conservative normalizer without changing their public interfaces.
_metadata.crossref_date = _safe_crossref_date
