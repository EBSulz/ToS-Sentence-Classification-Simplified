from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=200)


class SearchResult(BaseModel):
    sentence: str
    doc: str
    score: float
    is_unfair: bool


class SearchResponse(BaseModel):
    results: list[SearchResult]
    embedding_method: str


class ClassifyRequest(BaseModel):
    sentence: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    is_unfair: bool
    categories: list[str]   # e.g. ["ltd", "ter"]
    details: dict[str, bool]  # all 8 categories: {cat: True/False}


class AnalyzeRequest(BaseModel):
    sentence: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=200)


class AnalyzeResponse(BaseModel):
    classification: ClassifyResponse
    similar: list[SearchResult]


class HealthResponse(BaseModel):
    status: str
    embedding_method: str
    corpus_size: int
