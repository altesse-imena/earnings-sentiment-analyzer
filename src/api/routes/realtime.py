from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.services.price_stream import manager

router = APIRouter()


@router.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
