"""FastAPI entry point for recorded and server-approved RCA investigations."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings
from backend.app.schemas import (
    CaseDetail,
    CaseListResponse,
    CaseReadiness,
    HealthResponse,
    InvestigationRunView,
    RunCreateRequest,
    RunCreateResponse,
    RunListResponse,
    RunSummary,
)
from backend.app.services.case_catalog import (
    CaseCatalogUnavailable,
    CaseNotFound,
    Rca100CaseCatalog,
)
from backend.app.services.run_executor import RunJobManager, RunProfileUnavailable
from backend.app.services.run_projector import RunProjector
from backend.app.services.run_store import RunNotFound, RunStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_environment()
    manager = RunJobManager(
        settings.cases_root,
        settings.runs_root,
        max_workers=settings.live_workers,
        settings=settings,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="RCA Agent Investigation API",
        version="0.1.0",
        description=(
            "GT-free API for recorded and server-approved RCA100 investigations. "
            "It never exposes evaluator data or changes Core decisions."
        ),
        lifespan=lifespan,
    )
    # The console is served by Vite on a separate local port during development.
    # Accept local browser origins without coupling the API to one fixed Vite port.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.case_catalog = Rca100CaseCatalog(settings.cases_root)
    app.state.run_store = RunStore(settings.runs_root)
    app.state.projector = RunProjector()
    app.state.job_manager = manager

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        catalog = _catalog(request)
        try:
            catalog.list_cases()
            available = True
        except CaseCatalogUnavailable:
            available = False
        return HealthResponse(status="ok", cases_available=available)

    @app.get("/api/v1/cases", response_model=CaseListResponse)
    def list_cases(request: Request, query: str | None = None) -> CaseListResponse:
        catalog = _catalog(request)
        store = _store(request)
        try:
            cases = catalog.list_cases()
        except CaseCatalogUnavailable as exc:
            raise HTTPException(status_code=503, detail="RCA100 cases are unavailable") from exc
        counts = _run_counts(store)
        needle = query.casefold().strip() if query else ""
        items = []
        for item in cases:
            searchable = f"{item.case_id} {item.title} {item.alert_entity}".casefold()
            if needle and needle not in searchable:
                continue
            items.append(item.model_copy(update={"run_count": counts.get(item.case_id, 0)}))
        return CaseListResponse(items=tuple(items), total=len(items))

    @app.get("/api/v1/cases/{case_id}", response_model=CaseDetail)
    def get_case(case_id: str, request: Request) -> CaseDetail:
        try:
            item = _catalog(request).get_case(case_id)
        except CaseCatalogUnavailable as exc:
            raise HTTPException(status_code=503, detail="RCA100 cases are unavailable") from exc
        except CaseNotFound as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc
        count = sum(1 for run in _store(request).list_runs(case_id) if run.case_id == case_id)
        return item.model_copy(update={"run_count": count})

    @app.get("/api/v1/cases/{case_id}/runs", response_model=RunListResponse)
    def list_case_runs(case_id: str, request: Request) -> RunListResponse:
        # Verify the Case first so typoed IDs cannot look like empty histories.
        try:
            _catalog(request).get_case(case_id)
        except CaseCatalogUnavailable as exc:
            raise HTTPException(status_code=503, detail="RCA100 cases are unavailable") from exc
        except CaseNotFound as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc
        items = _store(request).list_runs(case_id)
        return RunListResponse(items=items, total=len(items))

    @app.post(
        "/api/v1/cases/{case_id}/runs",
        response_model=RunCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def create_run(
        case_id: str, request_body: RunCreateRequest, request: Request
    ) -> RunCreateResponse:
        try:
            case = _catalog(request).get_case(case_id)
        except CaseCatalogUnavailable as exc:
            raise HTTPException(status_code=503, detail="RCA100 cases are unavailable") from exc
        except CaseNotFound as exc:
            raise HTTPException(status_code=404, detail="Case not found") from exc
        if case.readiness != CaseReadiness.READY:
            raise HTTPException(status_code=409, detail="Case data is not ready for investigation")
        try:
            return _manager(request).submit(case_id, request_body.profile_id)
        except RunProfileUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail="The selected run profile is not configured on this host",
            ) from exc

    @app.get("/api/v1/runs/{run_id}/status", response_model=RunSummary)
    def get_run_status(run_id: str, request: Request) -> RunSummary:
        try:
            return _store(request).get_summary(run_id)
        except RunNotFound as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/v1/runs/{run_id}", response_model=InvestigationRunView)
    def get_run(run_id: str, request: Request) -> InvestigationRunView:
        try:
            stored = _store(request).get_run(run_id)
            case = _catalog(request).get_case(stored.manifest.case_id)
        except CaseCatalogUnavailable as exc:
            raise HTTPException(status_code=503, detail="RCA100 cases are unavailable") from exc
        except (RunNotFound, CaseNotFound) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        return _projector(request).project(stored, case)

    return app


def _catalog(request: Request) -> Rca100CaseCatalog:
    return request.app.state.case_catalog


def _store(request: Request) -> RunStore:
    return request.app.state.run_store


def _projector(request: Request) -> RunProjector:
    return request.app.state.projector


def _manager(request: Request) -> RunJobManager:
    return request.app.state.job_manager


def _run_counts(store: RunStore) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in store.list_runs():
        counts[run.case_id] = counts.get(run.case_id, 0) + 1
    return counts


app = create_app()
