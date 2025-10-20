from fastapi import APIRouter

from sherloque.search_engine import QueryEngine, Ranker, ScoreSuite
from app.models import QueryResponse

router = APIRouter(prefix="/query")

s = ScoreSuite()
r = Ranker(score_suite=s)
q = QueryEngine(ranker=r)

@router.get("", response_model=QueryResponse)
async def query_index(query: str) -> QueryResponse:
    return QueryResponse(
        results=await q.query(query),
        # results=await q.get_matched_urls(query),
    )
