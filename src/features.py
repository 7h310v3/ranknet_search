"""
src/features.py - Trích xuất đặc trưng cho xếp hạng
"""

import numpy as np
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

from .model import Document, Query


class FeatureExtractor:
    """Trích xuất đặc trưng cho các cặp truy vấn-tài liệu"""
    
    def __init__(self):
        self.feature_names = [
            # Đặc trưng khớp văn bản
            'bm25_score', 'tfidf_score', 'title_exact_match',
            'title_word_overlap', 'content_word_overlap',
            # Đặc trưng chất lượng tài liệu  
            'doc_length_norm', 'freshness', 'url_depth',
            # Đặc trưng phổ biến
            'views_norm', 'clicks_norm', 'ctr',
            # Đặc trưng nhấp chuột
            'dwell_time_norm', 'bounce_rate'
        ]
        self.n_features = len(self.feature_names)
    
    def extract(self, query: Query, doc: Document, 
                corpus_stats: Dict = None, 
                click_stats: Dict = None) -> np.ndarray:
        """
        Trích xuất tất cả đặc trưng cho một cặp truy vấn-tài liệu
        
        Tham số:
            query: Truy vấn tìm kiếm
            doc: Tài liệu
            corpus_stats: Thống kê kho dữ liệu cho IDF, v.v.
            click_stats: Thống kê nhấp chuột
            
        Trả về:
            Vector đặc trưng
        """
        features = []
        query_terms = query.text.lower().split()
        
        # Text matching features
        text_features = self._extract_text_features(query_terms, doc, corpus_stats)
        features.extend(text_features)
        
        # Document quality features
        quality_features = self._extract_quality_features(doc)
        features.extend(quality_features)
        
        # Popularity features
        popularity_features = self._extract_popularity_features(doc)
        features.extend(popularity_features)
        
        # Click features
        click_features = self._extract_click_features(doc, click_stats)
        features.extend(click_features)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_text_features(self, query_terms: List[str], 
                               doc: Document, 
                               corpus_stats: Dict) -> List[float]:
        """Extract text-based features"""
        features = []
        
        # BM25 score
        bm25 = self._calculate_bm25(query_terms, doc, corpus_stats)
        features.append(bm25)
        
        # TF-IDF score
        tfidf = self._calculate_tfidf(query_terms, doc, corpus_stats)
        features.append(tfidf)
        
        # Title exact match
        query_text = ' '.join(query_terms)
        title_exact = 1.0 if query_text in doc.title.lower() else 0.0
        features.append(title_exact)
        
        # Title word overlap
        title_words = set(doc.title.lower().split())
        title_overlap = len(set(query_terms) & title_words) / max(len(query_terms), 1)
        features.append(title_overlap)
        
        # Content word overlap
        content_words = set(doc.content.lower().split())
        content_overlap = len(set(query_terms) & content_words) / max(len(query_terms), 1)
        features.append(content_overlap)
        
        return features
    
    def _extract_quality_features(self, doc: Document) -> List[float]:
        """Extract document quality features"""
        features = []
        
        # Document length (normalized)
        doc_length = len(doc.content.split()) / 1000.0
        features.append(min(doc_length, 1.0))
        
        # Freshness
        if isinstance(doc.timestamp, datetime):
            days_old = (datetime.now() - doc.timestamp).days
            freshness = np.exp(-days_old / 365.0)
        else:
            freshness = 0.5
        features.append(freshness)
        
        # URL depth
        url_depth = doc.url.count('/') / 10.0
        features.append(min(url_depth, 1.0))
        
        return features
    
    def _extract_popularity_features(self, doc: Document) -> List[float]:
        """Extract popularity features"""
        features = []
        
        # Views (log normalized)
        views_norm = np.log1p(doc.views) / 15.0
        features.append(min(views_norm, 1.0))
        
        # Clicks (log normalized)
        clicks_norm = np.log1p(doc.clicks) / 10.0
        features.append(min(clicks_norm, 1.0))
        
        # CTR
        ctr = doc.clicks / max(doc.views, 1)
        features.append(min(ctr, 1.0))
        
        return features
    
    def _extract_click_features(self, doc: Document, 
                                click_stats: Dict) -> List[float]:
        """Extract click-based features"""
        features = []
        
        # Average dwell time (normalized to 5 minutes)
        dwell_norm = doc.dwell_time / 300.0
        features.append(min(dwell_norm, 1.0))
        
        # Bounce rate (simplified)
        bounce_rate = 1.0 - min(doc.dwell_time / 30.0, 1.0)
        features.append(bounce_rate)
        
        return features
    
    def _calculate_bm25(self, query_terms: List[str], 
                       doc: Document, 
                       corpus_stats: Dict) -> float:
        """Calculate BM25 score"""
        if not corpus_stats:
            return 0.0
        
        k1, b = 1.2, 0.75
        doc_words = doc.content.lower().split() + doc.title.lower().split()
        doc_len = len(doc_words)
        avg_doc_len = corpus_stats.get('avg_doc_len', 500)
        
        # Ensure avg_doc_len is not zero to avoid division by zero
        if avg_doc_len == 0:
            avg_doc_len = 500  # Use default value
            
        n_docs = corpus_stats.get('n_docs', 1000)
        
        score = 0.0
        for term in query_terms:
            tf = doc_words.count(term)
            df = corpus_stats.get('df', {}).get(term, 1)
            
            idf = np.log((n_docs - df + 0.5) / (df + 0.5))
            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
            score += idf * tf_component
        
        return score / (len(query_terms) + 1)
    
    def _calculate_tfidf(self, query_terms: List[str], 
                        doc: Document, 
                        corpus_stats: Dict) -> float:
        """Calculate TF-IDF score"""
        if not corpus_stats:
            return 0.0
        
        doc_words = doc.content.lower().split() + doc.title.lower().split()
        n_docs = corpus_stats.get('n_docs', 1000)
        
        score = 0.0
        for term in query_terms:
            tf = doc_words.count(term) / max(len(doc_words), 1)
            df = corpus_stats.get('df', {}).get(term, 1)
            idf = np.log(n_docs / (df + 1))
            score += tf * idf
        
        return score / (len(query_terms) + 1)