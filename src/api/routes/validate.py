from typing import Any

from fastapi import APIRouter, Depends

from src.api.deps import get_validation_service
from src.api.middleware.auth import require_auth
from src.rag.pipeline.validation import ValidationService
from src.schemas.api import ValidateRequest, ValidateResponse

router = APIRouter(prefix="/validate", tags=["validate"])


@router.post("", response_model=ValidateResponse)
async def validate(
    request: ValidateRequest,
    validation: ValidationService = Depends(get_validation_service),
    _user: dict[str, Any] = Depends(require_auth),
) -> ValidateResponse:
    """
    Check uploaded session documents against the normative_base for CBR compliance.
    """
    document_id = str(request.document_id) if request.document_id else None
    return await validation.validate(
        session_id=request.session_id,
        document_id=document_id,
    )
