from fastapi import APIRouter

from sherloque.search_engine import QueryEngine
from sherloque.search_engine.ranker import Ranker, ScoreSuite
from app.models import QueryResponse
from app.config import get_async_engine, get_settings

router = APIRouter(prefix="/query")

settings = get_settings()
engine = get_async_engine()

s = ScoreSuite()
r = Ranker(score_suite=s, engine=engine)
q = QueryEngine(ranker=r, engine=engine)

@router.get("", response_model=QueryResponse)
async def query_index(query: str) -> QueryResponse:
    return QueryResponse(
        results=await q.query(query),
        # results=await q.get_matched_urls(query),
    )
