import math

from fastapi import APIRouter, Request

router = APIRouter()


def _sanitize(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@router.get("/model/report")
def get_model_report(request: Request):
    return _sanitize(request.app.state.report)
