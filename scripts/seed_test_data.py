"""
Seeds sample vectors into all 4 Qdrant collections.
Loads multilingual-e5-large, embeds real texts, upserts real 1024-dim vectors.

Usage:
    python scripts/seed_test_data.py
    make seed-test-data

Note: First run downloads ~560MB model to ./models/
"""

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import configure_logging, get_logger
from src.rag.config import CollectionName, get_config
from src.rag.embeddings.e5_large import E5LargeEmbeddingProvider
from src.rag.vectorstore.client import QdrantManager
from src.rag.vectorstore.operations import VectorStoreOperations
from src.schemas.chunk import Chunk, ChunkMetadata
from src.schemas.document import DocType

SEED_DATA: dict[CollectionName, list[dict]] = {  # type: ignore[type-arg]
    CollectionName.NORMATIVE_BASE: [
        {
            "text": (
                "Положение Банка России 254-П. Кредитная организация обязана формировать резервы "
                "на возможные потери по ссудам в соответствии с требованиями настоящего Положения. "
                "Резерв формируется при обесценении ссуды вследствие неисполнения обязательств."
            ),
            "metadata": {
                "doc_type": DocType.NORMATIVE,
                "law_article": "254-П",
                "effective_date": "2004-03-26",
            },
        },
        {
            "text": (
                "Федеральный закон 102-ФЗ 'Об ипотеке'. Статья 1. По договору об ипотеке "
                "одна сторона — залогодержатель, являющийся кредитором по обязательству, "
                "обеспеченному ипотекой, имеет право получить удовлетворение своих денежных требований."
            ),
            "metadata": {
                "doc_type": DocType.NORMATIVE,
                "law_article": "102-ФЗ-ст1",
                "effective_date": "1998-07-16",
            },
        },
    ],
    CollectionName.DEAL_PRECEDENTS: [
        {
            "text": (
                "Кредитная сделка ID: DEAL-2023-001. Заёмщик: ООО 'Альфа-Строй'. "
                "Обеспечение: залог недвижимости, рыночная стоимость 150 млн руб. "
                "Исход: одобрено. Замечания: страхование объекта залога подтверждено."
            ),
            "metadata": {
                "doc_type": DocType.CONTRACT,
                "deal_id": "DEAL-2023-001",
                "outcome": "approved",
                "risk_flags": [],
            },
        },
        {
            "text": (
                "Кредитная сделка ID: DEAL-2023-045. Заёмщик: ИП Петров И.А. "
                "Обеспечение: залог оборудования. Исход: отклонено. "
                "Причина: недостаточная ликвидность залога, износ >70%."
            ),
            "metadata": {
                "doc_type": DocType.CONTRACT,
                "deal_id": "DEAL-2023-045",
                "outcome": "rejected",
                "risk_flags": ["low_liquidity", "high_depreciation"],
            },
        },
    ],
    CollectionName.REFERENCE_TEMPLATES: [
        {
            "text": (
                "Эталон договора ипотеки v2.1. Преамбула: настоящий договор заключён между "
                "Банком (залогодержатель) и Заёмщиком (залогодатель) в соответствии с ГК РФ "
                "и ФЗ-102. Предмет залога: жилое помещение, кадастровый номер [КАДАСТР]."
            ),
            "metadata": {
                "doc_type": DocType.TEMPLATE,
                "template_version": "2.1",
                "section": "preamble",
            },
        },
    ],
    CollectionName.CURRENT_PACKAGE: [
        {
            "text": (
                "Договор залога недвижимости. Стороны: ПАО 'Банк' и ООО 'Заёмщик'. "
                "Предмет: нежилое помещение площадью 450 кв.м, г. Москва, ул. Ленина, 10. "
                "Оценочная стоимость: 85 000 000 рублей."
            ),
            "metadata": {
                "doc_type": DocType.CONTRACT,
                "session_id": "test-session-001",
                "page": 1,
            },
        },
    ],
}


async def main() -> int:
    config = get_config()
    configure_logging(config.rag_log_level)
    logger = get_logger("seed_test_data")

    # Initialize embedding model
    logger.info("loading e5-large model (first run downloads ~560MB)...")
    embedder = E5LargeEmbeddingProvider(config)
    await embedder.initialize()

    # Initialize Qdrant
    manager = QdrantManager(config)
    try:
        await manager.initialize()
    except Exception as exc:
        logger.error("failed to connect to Qdrant", error=str(exc))
        return 1

    ops = VectorStoreOperations(manager, config)

    total_upserted = 0
    for collection, items in SEED_DATA.items():
        texts = [item["text"] for item in items]
        embeddings = await embedder.embed_texts(texts)

        chunks: list[Chunk] = []
        for item, embedding in zip(items, embeddings, strict=True):
            doc_id = uuid.uuid4()
            meta = item["metadata"]
            metadata = ChunkMetadata(
                document_id=doc_id,
                doc_type=meta["doc_type"],
                chunk_index=0,
                total_chunks=1,
                law_article=meta.get("law_article"),
                effective_date=meta.get("effective_date"),
                deal_id=meta.get("deal_id"),
                outcome=meta.get("outcome"),
                risk_flags=meta.get("risk_flags", []),
                template_version=meta.get("template_version"),
                section=meta.get("section"),
                session_id=meta.get("session_id"),
                page=meta.get("page"),
            )
            chunk = Chunk(
                text=item["text"],
                token_count=len(item["text"].split()),
                metadata=metadata,
                embedding=embedding,
            )
            chunks.append(chunk)

        count = await ops.upsert_chunks(collection, chunks)
        total_upserted += count
        logger.info("seeded collection", collection=collection, count=count)

    await manager.close()
    logger.info("seed complete", total_upserted=total_upserted)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
