from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class MatchMethod(str, Enum):
    classic = "classic"
    dl = "dl"
    dedicated = "dedicated"


class OverlayMatch(BaseModel):
    a: Tuple[float, float]
    b: Tuple[float, float]
    kind: str = Field(..., description="tentative|inlier|outlier")
    sim: Optional[float] = None  


class Overlay(BaseModel):
    matches: List[OverlayMatch] = Field(default_factory=list)


class MatchResponse(BaseModel):
    method: MatchMethod
    score: float
    decision: bool
    threshold: float
    latency_ms: float
    meta: Dict[str, Any] = Field(default_factory=dict)
    overlay: Optional[Overlay] = None
