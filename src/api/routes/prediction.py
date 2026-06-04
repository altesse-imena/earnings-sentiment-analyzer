from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/prediction/{ticker}/{date}")
def get_prediction(ticker: str, date: str, request: Request):
    model = request.app.state.model
    df_matrix = request.app.state.df_matrix
    report = request.app.state.report
    shap_df = request.app.state.shap_df

    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded — run training first")
    if df_matrix.empty or not report:
        raise HTTPException(status_code=404, detail="Feature matrix not available")

    fc = report.get("feature_columns", [])
    mask = (df_matrix["ticker"] == ticker) & (df_matrix["event_date"] == date)
    match = df_matrix[mask]

    if match.empty or not fc:
        raise HTTPException(status_code=404, detail=f"No feature data for {ticker} on {date}")

    X_input = match[fc].fillna(0).iloc[[0]]
    prob = model.predict_proba(X_input)[0]
    pred_label = int(model.predict(X_input)[0])
    confidence = float(prob[pred_label])

    actual_change = None
    if "pct_change_48h" in match.columns:
        try:
            actual_change = round(float(match["pct_change_48h"].iloc[0]), 4)
        except (TypeError, ValueError):
            pass

    top_feature = None
    if not shap_df.empty:
        top_row = shap_df.sort_values("mean_abs_shap", ascending=False).iloc[0]
        top_feature = str(top_row["feature"])

    return {
        "label": pred_label,
        "direction": "UP" if pred_label == 1 else "DOWN",
        "confidence": round(confidence, 4),
        "actual_change": actual_change,
        "top_feature": top_feature,
    }
