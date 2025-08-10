"""
src/models.py - Data models for RankNet search engine
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np


@dataclass
class Document:
    """Document in search corpus"""
    doc_id: str
    title: str
    content: str
    url: str
    author: str
    timestamp: datetime
    views: int = 0
    clicks: int = 0
    dwell_time: float = 0.0
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'author': self.author,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'views': self.views,
            'clicks': self.clicks,
            'dwell_time': self.dwell_time,
            'tags': self.tags
        }


@dataclass
class Query:
    """Search query"""
    query_id: str
    text: str
    user_id: str
    timestamp: datetime
    clicked_docs: List[str] = field(default_factory=list)
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'query_id': self.query_id,
            'text': self.text,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'clicked_docs': self.clicked_docs,
            'relevance_scores': self.relevance_scores
        }


@dataclass
class PairwiseExample:
    """Training example for pairwise ranking"""
    query_id: str
    doc_i_id: str  # Higher relevance
    doc_j_id: str  # Lower relevance
    features_i: np.ndarray
    features_j: np.ndarray
    label: float  # 1.0 if doc_i > doc_j
    weight: float = 1.0


@dataclass
class ClickLog:
    """User click interaction"""
    query_id: str
    doc_id: str
    position: int
    dwell_time: float
    timestamp: datetime
    user_id: str