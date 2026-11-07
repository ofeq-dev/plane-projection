"""Shared error types for plane-projection."""

from __future__ import annotations


class UsageError(Exception):
    """A user-fixable input/usage problem reported as a one-line CLI error (FR-2)."""