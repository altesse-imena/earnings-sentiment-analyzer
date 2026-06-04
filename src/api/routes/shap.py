from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/shap")
def get_shap(request: Request):
    shap_df = request.app.state.shap_df
    if shap_df.empty:
        return {"features": []}

    top8 = shap_df.nlargest(8, "mean_abs_shap")
    return {
        "features": [
            {
                "feature": str(row["feature"]),
                "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
            }
            for _, row in top8.iterrows()
        ]
    }
