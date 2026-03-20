from src.common.logging import get_logger
from src.rag.config import CollectionName

logger = get_logger(__name__)

# Keywords that signal the query targets a specific collection
_NORMATIVE_KW = {
    "закон", "статья", "норматив", "цб", "положение", "указание",
    "инструкция", "фз", "постановление", "регулирование", "нормативный",
    "центробанк", "банка", "требование", "соответствие", "регулятор",
}
_PRECEDENT_KW = {
    "сделка", "прецедент", "дело", "исход", "одобрен", "одобрить",
    "отказ", "кейс", "аналог", "история", "решение", "прошлый", "предыдущий",
    "похожий", "аналогичный",
}
_TEMPLATE_KW = {
    "шаблон", "образец", "форма", "эталон", "типовой", "типовая",
    "пример", "образцовый", "стандартный",
}


class QueryRouter:
    """
    Routes a query to the appropriate Qdrant collection.

    Rules (in priority order):
    1. session_id present → current_package (user uploaded documents)
    2. Normative keywords → normative_base
    3. Precedent keywords → deal_precedents
    4. Template keywords → reference_templates
    5. Default → current_package
    """

    def route(
        self,
        query: str,
        session_id: str | None = None,
    ) -> CollectionName:
        if session_id:
            return CollectionName.CURRENT_PACKAGE

        words = set(query.lower().split())

        if words & _NORMATIVE_KW:
            collection = CollectionName.NORMATIVE_BASE
        elif words & _PRECEDENT_KW:
            collection = CollectionName.DEAL_PRECEDENTS
        elif words & _TEMPLATE_KW:
            collection = CollectionName.REFERENCE_TEMPLATES
        else:
            collection = CollectionName.CURRENT_PACKAGE

        logger.debug("query routed", collection=collection, query_preview=query[:60])
        return collection
