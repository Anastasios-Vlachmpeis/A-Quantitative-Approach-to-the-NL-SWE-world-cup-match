"""Pandera schema registry for canonical and feature tables (filled in Plans 03–04)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera as pa

if TYPE_CHECKING:
    import pandas as pd

# Registry: schema_name -> pandera DataFrameSchema (populated in later plans)
SCHEMA_REGISTRY: dict[str, pa.DataFrameSchema] = {}


def register_schema(name: str, schema: pa.DataFrameSchema) -> None:
    """Register a named schema for validation."""
    SCHEMA_REGISTRY[name] = schema


def validate_table(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """Validate df against a registered schema; raise if unknown or invalid."""
    if schema_name not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown schema: {schema_name!r}. Registered: {list(SCHEMA_REGISTRY)}")
    return SCHEMA_REGISTRY[schema_name].validate(df)
