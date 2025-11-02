import time

from fastapi import FastAPI, Request

from app.config import get_async_lifespan
from app.routes.api_routes import router as api_router

app = FastAPI(
    lifespan=get_async_lifespan(),
)

# --- routers ---
app.include_router(api_router)


# === routers ===

# -- middlewares ---
@app.middleware("http")
async def log_wait_time_middleware(request: Request, call_next):
    """
    This middleware calculates the time a request spent in the network and
    the server's queue before being processed.
    """
    start_time_header = request.headers.get("X-Request-Start-Time")
    if start_time_header:
        process_start_time = time.time()
        client_sent_time = float(start_time_header)
        wait_time = process_start_time - client_sent_time
        print(f"Request to {request.url.path} waited {wait_time:.4f} seconds.")
    response = await call_next(request)
    return response


# === middlewares ===

if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True, workers=4)
