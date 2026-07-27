"""User-facing exception formatting without exposing internal tracebacks."""

from __future__ import annotations

import logging


def user_error(context: str, exception: BaseException) -> str:
    logging.getLogger("voxis").exception(
        "%s failed", context, exc_info=exception
    )
    detail = str(exception).strip() or exception.__class__.__name__
    return f"{context}: {detail}"
