from fastapi import APIRouter
from .query_routes import router as query_router

router = APIRouter(prefix="/v1")
router.include_router(query_router)
