from enum import Enum

from pydantic import BaseModel, Field


class QueryRoute(str, Enum):
    VECTOR = "VECTOR"
    GRAPH = "GRAPH"


class RouteDecision(BaseModel):
    route: QueryRoute
    reason: str = Field(min_length=1)