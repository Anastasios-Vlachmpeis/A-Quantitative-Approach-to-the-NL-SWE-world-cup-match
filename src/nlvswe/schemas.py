"""Pandera schemas for canonical and feature tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera as pa
from pandera import Check

if TYPE_CHECKING:
    import pandas as pd

SCHEMA_REGISTRY: dict[str, pa.DataFrameSchema] = {}

TEAMS_SCHEMA = pa.DataFrameSchema(
    {
        "team_id": pa.Column(str, unique=True, nullable=False),
        "canonical_name": pa.Column(str, nullable=False),
        "country_code": pa.Column(str, nullable=True, required=False),
        "confederation": pa.Column(str, nullable=True, required=False),
        "aliases": pa.Column(object, nullable=False),
    },
    strict=True,
    name="teams",
)

VENUES_SCHEMA = pa.DataFrameSchema(
    {
        "venue_id": pa.Column(str, unique=True, nullable=False),
        "name": pa.Column(str, nullable=False),
        "city": pa.Column(str, nullable=True, required=False),
        "country_code": pa.Column(str, nullable=True, required=False),
        "lat": pa.Column(float, nullable=True, required=False),
        "lon": pa.Column(float, nullable=True, required=False),
        "altitude_m": pa.Column(float, nullable=True, required=False),
        "capacity": pa.Column("Int64", nullable=True, required=False),
    },
    strict=True,
    name="venues",
)

MATCHES_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, unique=True, nullable=False),
        "date_utc": pa.Column("datetime64[ns, UTC]", nullable=False),
        "competition": pa.Column(str, nullable=False),
        "season": pa.Column(str, nullable=True, required=False),
        "stage": pa.Column(str, nullable=True, required=False),
        "home_team_id": pa.Column(str, nullable=False),
        "away_team_id": pa.Column(str, nullable=False),
        "venue_id": pa.Column(str, nullable=True, required=False),
        "neutral": pa.Column(bool, nullable=False),
        "home_goals": pa.Column("Int64", Check.ge(0), nullable=True, required=False),
        "away_goals": pa.Column("Int64", Check.ge(0), nullable=True, required=False),
        "status": pa.Column(str, Check.isin(["completed", "scheduled"]), nullable=False),
        "went_to_shootout": pa.Column("boolean", nullable=True, required=False),
    },
    strict=True,
    name="matches",
)

RATINGS_SCHEMA = pa.DataFrameSchema(
    {
        "team_id": pa.Column(str, nullable=False),
        "rating_date": pa.Column("datetime64[ns, UTC]", nullable=False),
        "source": pa.Column(str, Check.isin(["fifa", "elo"]), nullable=False),
        "value": pa.Column(float, nullable=False),
        "rank": pa.Column("Int64", nullable=True, required=False),
    },
    strict=True,
    name="ratings",
)

TEAM_MATCH_STATS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "team_id": pa.Column(str, nullable=False),
        "xg": pa.Column(float, nullable=True, required=False),
        "shots": pa.Column("Int64", nullable=True, required=False),
        "possession": pa.Column(float, nullable=True, required=False),
    },
    strict=True,
    name="team_match_stats",
)

ODDS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "bookmaker": pa.Column(str, nullable=False),
        "market": pa.Column(str, Check.isin(["1x2", "totals", "ah"]), nullable=False),
        "selection": pa.Column(str, nullable=False),
        "decimal_odds": pa.Column(float, Check.gt(1.0), nullable=False),
        "captured_at": pa.Column("datetime64[ns, UTC]", nullable=False),
        "is_closing": pa.Column(bool, nullable=False),
    },
    strict=True,
    name="odds",
)

CONDITIONS_SCHEMA = pa.DataFrameSchema(
    {
        "match_id": pa.Column(str, nullable=False),
        "kickoff_local": pa.Column("datetime64[ns, UTC]", nullable=True, required=False),
        "temp_c": pa.Column(float, nullable=True, required=False),
        "humidity": pa.Column(float, nullable=True, required=False),
        "weather": pa.Column(str, nullable=True, required=False),
        "altitude_m": pa.Column(float, nullable=True, required=False),
    },
    strict=True,
    name="conditions",
)

CANONICAL_SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "teams": TEAMS_SCHEMA,
    "venues": VENUES_SCHEMA,
    "matches": MATCHES_SCHEMA,
    "ratings": RATINGS_SCHEMA,
    "team_match_stats": TEAM_MATCH_STATS_SCHEMA,
    "odds": ODDS_SCHEMA,
    "conditions": CONDITIONS_SCHEMA,
}

for _name, _schema in CANONICAL_SCHEMAS.items():
    SCHEMA_REGISTRY[_name] = _schema


def register_schema(name: str, schema: pa.DataFrameSchema) -> None:
    """Register a named schema for validation."""
    SCHEMA_REGISTRY[name] = schema


def validate_table(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """Validate df against a registered schema; raise if unknown or invalid."""
    if schema_name not in SCHEMA_REGISTRY:
        raise KeyError(f"Unknown schema: {schema_name!r}. Registered: {list(SCHEMA_REGISTRY)}")
    return SCHEMA_REGISTRY[schema_name].validate(df)


def validate_business_rules(matches: pd.DataFrame, odds: pd.DataFrame) -> list[str]:
    """Return list of business-rule violations (empty = pass)."""
    issues: list[str] = []

    if not matches.empty:
        self_play = matches[matches["home_team_id"] == matches["away_team_id"]]
        if len(self_play):
            issues.append(f"matches: {len(self_play)} rows with home_team_id == away_team_id")

        dupes = matches["match_id"].duplicated()
        if dupes.any():
            issues.append(f"matches: {dupes.sum()} duplicate match_id values")

        completed = matches[matches["status"] == "completed"]
        bad_scores = completed[
            completed["home_goals"].isna()
            | completed["away_goals"].isna()
            | (completed["home_goals"] < 0)
            | (completed["away_goals"] < 0)
        ]
        if len(bad_scores):
            issues.append(f"matches: {len(bad_scores)} completed matches with invalid scores")

    if not odds.empty:
        bad_odds = odds[odds["decimal_odds"] <= 1.0]
        if len(bad_odds):
            issues.append(f"odds: {len(bad_odds)} rows with decimal_odds <= 1.0")

        one_x_two = odds[odds["market"] == "1x2"]
        if not one_x_two.empty:
            for (mid, bk), grp in one_x_two.groupby(["match_id", "bookmaker"]):
                closing = grp[grp["is_closing"]]
                if closing.empty:
                    continue
                implied = (1.0 / closing["decimal_odds"]).sum()
                if implied < 0.95 or implied > 1.25:
                    issues.append(
                        f"odds: 1x2 overround {implied:.3f} out of band for match={mid} book={bk}"
                    )
                    break

    return issues
