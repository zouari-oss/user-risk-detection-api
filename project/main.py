from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import health, predict

app = FastAPI(
    title="User Risk Detection API",
    description="Detect user vulnerabilities risk",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"details": "Welcome to user-risk-detection-api!"}


app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
