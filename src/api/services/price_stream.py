import asyncio
from datetime import datetime

import yfinance as yf
from fastapi import WebSocket

WATCH_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


async def price_broadcast_loop():
    prev_close: dict[str, float] = {}
    while True:
        try:
            payload: dict[str, dict] = {}
            for ticker in WATCH_TICKERS:
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="2d", interval="1m")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                        midpoint = max(len(hist) // 2, 1)
                        prev = prev_close.get(ticker, float(hist["Close"].iloc[-midpoint]))
                        change = round(price - prev, 2)
                        change_pct = round((price - prev) / prev * 100, 2) if prev else 0.0
                        payload[ticker] = {
                            "price": round(price, 2),
                            "change": change,
                            "change_pct": change_pct,
                        }
                        prev_close[ticker] = float(hist["Close"].iloc[-midpoint])
                except Exception:
                    pass

            if payload:
                await manager.broadcast(
                    {
                        "type": "price_update",
                        "timestamp": datetime.now().isoformat(),
                        "prices": payload,
                    }
                )
        except Exception:
            pass

        await asyncio.sleep(30)
