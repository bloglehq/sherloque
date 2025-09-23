from fastapi import FastAPI

from routes.api_routes import router as api_router

app = FastAPI()

# --- routers ---
app.include_router(api_router)

# === routers ===


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True, workers=4)
