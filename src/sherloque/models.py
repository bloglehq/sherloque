from typing import Any

from pydantic import BaseModel


class URLMatch(BaseModel):
    url: str
    url_id: int


class URLMatchDetailModel(BaseModel):
    url_matches: list[URLMatch]
    token_ids: list[Any]


class Score(BaseModel):
    score: float
    url: str


class ScoreDetailModel(BaseModel):
    name: str
    query: str
    weighted_scores: list[Score]


__all__ = [
    "URLMatch",
    "Score",
    "ScoreDetailModel",
]
