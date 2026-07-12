"""Versioned local annotation tooling for RKHunter."""

from .. import __version__ as TOOL_VERSION

ANNOTATION_SCHEMA_VERSION = 2
DATABASE_SCHEMA_VERSION = 2

__all__ = ["ANNOTATION_SCHEMA_VERSION", "DATABASE_SCHEMA_VERSION", "TOOL_VERSION"]
