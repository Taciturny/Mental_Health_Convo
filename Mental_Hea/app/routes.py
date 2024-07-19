from fastapi import APIRouter
from app.models.response_generator import ResponseGenerator
from app.services.weaviate_schema import WeaviateService


router = APIRouter()

response_generator = ResponseGenerator()
weaviate_service = WeaviateService()

@router.get("/search")
async def search_endpoint(query: str):
    response = response_generator.get_response(query, weaviate_service.client)
    return {"response": response}
