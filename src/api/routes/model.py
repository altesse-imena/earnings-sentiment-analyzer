from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/model/report")
def get_model_report(request: Request):
    return request.app.state.report
