from dataclasses import dataclass, field

from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    Modifier,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from src.rag.config import CollectionName

DENSE_VECTOR_NAME = "text"
SPARSE_VECTOR_NAME = "bm25"


@dataclass
class CollectionDefinition:
    name: CollectionName
    description: str
    # field_name → schema type: "keyword" | "integer" | "datetime" | "float"
    payload_indexes: dict[str, str] = field(default_factory=dict)


def get_dense_vector_params(vector_size: int = 1024) -> VectorParams:
    return VectorParams(
        size=vector_size,
        distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(
                type=ScalarType.INT8,
                always_ram=True,
            )
        ),
    )


def get_sparse_vector_params() -> SparseVectorParams:
    return SparseVectorParams(
        index=SparseIndexParams(on_disk=False),
        modifier=Modifier.IDF,
    )


COLLECTION_CONFIGS: dict[CollectionName, CollectionDefinition] = {
    CollectionName.NORMATIVE_BASE: CollectionDefinition(
        name=CollectionName.NORMATIVE_BASE,
        description="Laws, CBR regulations, bank internal instructions",
        payload_indexes={
            "doc_type": "keyword",
            "law_article": "keyword",
            "document_id": "keyword",
            "chunk_index": "integer",
        },
    ),
    CollectionName.DEAL_PRECEDENTS: CollectionDefinition(
        name=CollectionName.DEAL_PRECEDENTS,
        description="Closed credit deals from core banking system",
        payload_indexes={
            "doc_type": "keyword",
            "deal_id": "keyword",
            "outcome": "keyword",
            "document_id": "keyword",
            "chunk_index": "integer",
        },
    ),
    CollectionName.REFERENCE_TEMPLATES: CollectionDefinition(
        name=CollectionName.REFERENCE_TEMPLATES,
        description="Reference templates and few-shot examples",
        payload_indexes={
            "doc_type": "keyword",
            "template_version": "keyword",
            "document_id": "keyword",
            "chunk_index": "integer",
        },
    ),
    CollectionName.CURRENT_PACKAGE: CollectionDefinition(
        name=CollectionName.CURRENT_PACKAGE,
        description="Current session document package (TTL — deleted after session ends)",
        payload_indexes={
            "session_id": "keyword",
            "doc_type": "keyword",
            "document_id": "keyword",
            "page": "integer",
            "chunk_index": "integer",
        },
    ),
}
