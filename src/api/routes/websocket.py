import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.common.logging import get_logger
from src.rag.pipeline.retrieval import RetrievalService, _build_context
from src.schemas.api import WsChatMessage, WsCitation, WsDone, WsError, WsToken

router = APIRouter(tags=["websocket"])
logger = get_logger(__name__)


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """
    Streaming Q&A over WebSocket.

    Client sends: WsChatMessage JSON
    Server sends: WsCitation → WsToken* → WsDone  (or WsError on failure)
    """
    await websocket.accept()
    logger.info("ws connection opened", client=str(websocket.client))

    retrieval: RetrievalService = websocket.app.state.retrieval_service

    try:
        while True:
            raw = await websocket.receive_text()
            msg = WsChatMessage.model_validate_json(raw)
            t0 = time.perf_counter()

            try:
                # 1. Embed query and search
                query_vector = await retrieval._embedding.embed_query(msg.query)

                payload_filter = {}
                if msg.session_id:
                    from src.rag.config import CollectionName
                    if msg.collection == CollectionName.CURRENT_PACKAGE:
                        payload_filter["session_id"] = msg.session_id

                hits = await retrieval._vs.search(
                    collection=msg.collection,
                    query_vector=query_vector,
                    limit=5,
                    filter_payload=payload_filter or None,
                )

                # 2. Send citations immediately
                context, citations = _build_context(hits)
                await websocket.send_text(
                    WsCitation(citations=citations).model_dump_json()
                )

                # 3. Stream LLM tokens
                prompt = retrieval._qa_prompt.format(context=context, query=msg.query)
                async for token in retrieval._llm.stream(prompt):
                    await websocket.send_text(WsToken(content=token).model_dump_json())

                # 4. Done
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                await websocket.send_text(WsDone(latency_ms=latency_ms).model_dump_json())

                logger.info(
                    "ws chat completed",
                    session_id=msg.session_id,
                    latency_ms=latency_ms,
                )

            except Exception as exc:
                logger.error("ws chat error", error=str(exc))
                await websocket.send_text(WsError(message=str(exc)).model_dump_json())

    except WebSocketDisconnect:
        logger.info("ws connection closed", client=str(websocket.client))
