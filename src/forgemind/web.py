from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    mode: Literal["retrieve", "reason", "investigate"] = "reason"


def create_app(service: object) -> FastAPI:
    app = FastAPI(title="ForgeMind", docs_url=None, redoc_url=None)
    package_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=package_dir / "templates")
    app.mount("/static", StaticFiles(directory=package_dir / "static"), name="static")

    @app.get("/")
    def index(request: Request) -> object:
        return templates.TemplateResponse(request=request, name="index.html", context={})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/ask")
    def ask(request: AskRequest) -> dict[str, object]:
        return service.ask(request.question, request.mode).model_dump()

    return app
