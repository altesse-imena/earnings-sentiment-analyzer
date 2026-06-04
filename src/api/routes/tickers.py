from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/tickers")
def get_tickers(request: Request):
    df = request.app.state.df_features
    if df.empty:
        return {"tickers": []}
    return {"tickers": sorted(df["ticker"].unique().tolist())}


@router.get("/tickers/{ticker}/dates")
def get_dates(ticker: str, request: Request):
    df = request.app.state.df_features
    if df.empty:
        return {"dates": []}
    subset = df[df["ticker"] == ticker]["event_date"].unique().tolist()
    return {"dates": sorted(subset, reverse=True)}
