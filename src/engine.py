
"""
src/engine.py - Công cụ tìm kiếm chính với RankNet
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import pickle
import random

from .model import Document, Query, PairwiseExample, ClickLog
from .features import FeatureExtractor
from .ranknet import RankNet, RankNetTrainer
from .utils import compute_ndcg


class SearchEngine:
    """Công cụ tìm kiếm chính với xếp hạng RankNet"""
    
    def __init__(self, use_ranknet: bool = True):
        """
        Khởi tạo công cụ tìm kiếm
        
        Tham số:
            use_ranknet: Có sử dụng RankNet để xếp hạng hay không
        """
        # Lưu trữ tài liệu
        self.documents = {}  # doc_id -> Document
        self.document_index = defaultdict(set)  # term -> set of doc_ids
        
        # Các mô hình
        self.feature_extractor = FeatureExtractor()
        self.ranknet = None
        self.use_ranknet = use_ranknet
        
        # Thống kê
        self.corpus_stats = {}
        self.click_stats = defaultdict(dict)
        
        # Nhật ký
        self.query_log = []
        self.click_log = []
    
    def add_document(self, doc: Document):
        """Thêm tài liệu vào công cụ tìm kiếm"""
        self.documents[doc.doc_id] = doc
        self._update_index(doc)
        self._update_corpus_stats(doc)
    
    def add_documents_batch(self, docs: List[Document]):
        """Thêm nhiều tài liệu cùng lúc"""
        for doc in docs:
            self.add_document(doc)
        print(f"Đã thêm {len(docs)} tài liệu")
    
    def _update_index(self, doc: Document):
        """Cập nhật chỉ mục ngược"""
        terms = set(doc.title.lower().split() + doc.content.lower().split())
        for term in terms:
            self.document_index[term].add(doc.doc_id)
    
    def _update_corpus_stats(self, doc: Document):
        """Cập nhật thống kê kho dữ liệu"""
        if 'n_docs' not in self.corpus_stats:
            self.corpus_stats = {
                'n_docs': 0,
                'df': defaultdict(int),
                'total_doc_len': 0
            }
        
        self.corpus_stats['n_docs'] += 1
        
        # Cập nhật tần suất tài liệu
        doc_terms = set(doc.content.lower().split() + doc.title.lower().split())
        for term in doc_terms:
            self.corpus_stats['df'][term] += 1
        
        # Cập nhật độ dài trung bình của tài liệu
        doc_len = len(doc.content.split())
        self.corpus_stats['total_doc_len'] += doc_len
        self.corpus_stats['avg_doc_len'] = (
            self.corpus_stats['total_doc_len'] / self.corpus_stats['n_docs']
        )
    
    def load_data(self, filepath: str):
        """Tải tài liệu từ tệp CSV"""
        try:
            df = pd.read_csv(filepath)
            documents = []
            
            for idx, row in df.iterrows():
                doc = Document(
                    doc_id=str(idx),
                    title=str(row.get('title', '')),
                    content=str(row.get('text', row.get('content', ''))),
                    url=str(row.get('url', f'http://example.com/{idx}')),
                    author=str(row.get('author', row.get('authors', 'Unknown'))),
                    timestamp=pd.to_datetime(row.get('timestamp', datetime.now())),
                    views=int(row.get('claps', row.get('views', 0)) * 10),
                    clicks=int(row.get('claps', row.get('clicks', 0))),
                    tags=str(row.get('tags', '')).split('|') if 'tags' in row else []
                )
                documents.append(doc)
            
            self.add_documents_batch(documents)
            print(f"Đã tải {len(documents)} tài liệu từ {filepath}")
            
        except FileNotFoundError:
            print(f"Không tìm thấy tệp {filepath}. Sử dụng dữ liệu mẫu...")
            self.create_sample_data()
    
    def create_sample_data(self, n_docs: int = 100):
        """Tạo tài liệu mẫu cho việc kiểm thử"""
        topics = [
            'machine learning', 'deep learning', 'neural networks',
            'python programming', 'web development', 'data science',
            'artificial intelligence', 'natural language processing',
            'computer vision', 'reinforcement learning'
        ]
        
        documents = []
        for i in range(n_docs):
            topic = random.choice(topics)
            doc = Document(
                doc_id=f"doc_{i}",
                title=f"{topic.title()} Tutorial Part {i % 10 + 1}",
                content=f"This is a comprehensive guide about {topic}. " * 50,
                url=f"http://example.com/article/{i}",
                author=f"Author {i % 10}",
                timestamp=datetime.now() - pd.Timedelta(days=random.randint(1, 365)),
                views=random.randint(100, 10000),
                clicks=random.randint(10, 1000),
                tags=[topic.replace(' ', '_'), 'tutorial', 'tech']
            )
            documents.append(doc)
        
        self.add_documents_batch(documents)
    
    def initial_retrieval(self, query: Query, top_k: int = 100) -> List[str]:
        """Giai đoạn truy xuất ban đầu sử dụng chỉ mục ngược"""
        query_terms = set(query.text.lower().split())
        doc_scores = defaultdict(float)
        
        # Tính điểm TF-IDF đơn giản
        for term in query_terms:
            if term in self.document_index:
                df = len(self.document_index[term])
                idf = np.log((self.corpus_stats['n_docs'] + 1) / (df + 1))
                
                for doc_id in self.document_index[term]:
                    doc_scores[doc_id] += idf
        
        # Sắp xếp và trả về top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:top_k]]
    
    def rank_documents(self, query: Query, doc_ids: List[str]) -> List[Tuple[str, float]]:
        """Xếp hạng tài liệu sử dụng RankNet hoặc cách tính điểm dự phòng"""
        if not doc_ids:
            return []
        
        # Trích xuất đặc trưng cho tất cả tài liệu
        features_list = []
        valid_doc_ids = []
        
        for doc_id in doc_ids:
            if doc_id in self.documents:
                doc = self.documents[doc_id]
                features = self.feature_extractor.extract(
                    query, doc, self.corpus_stats, self.click_stats
                )
                features_list.append(features)
                valid_doc_ids.append(doc_id)
        
        if not features_list:
            return []
        
        # Tính điểm tài liệu
        if self.use_ranknet and self.ranknet:
            # Sử dụng RankNet
            features_array = np.array(features_list)
            scores = self.ranknet.predict(features_array)
        else:
            # Dự phòng: tổng có trọng số của đặc trưng
            features_array = np.array(features_list)
            # Tạo vector trọng số có cùng độ dài với đặc trưng
            n_features = features_array.shape[1]
            weights = np.ones(n_features) / n_features  # Trọng số bằng nhau
            scores = np.dot(features_array, weights)
        
        # Combine doc_ids with scores
        doc_scores = list(zip(valid_doc_ids, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores
    
    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Hàm tìm kiếm chính
        
        Tham số:
            query_text: Truy vấn tìm kiếm
            top_k: Số lượng kết quả
            
        Trả về:
            Kết quả tìm kiếm
        """
        # Tạo đối tượng truy vấn
        query = Query(
            query_id=f"q_{datetime.now().timestamp()}",
            text=query_text,
            user_id="user",
            timestamp=datetime.now()
        )
        
        # Truy xuất ban đầu
        candidate_docs = self.initial_retrieval(query, top_k=100)
        
        if not candidate_docs:
            return []
        
        # Xếp hạng
        ranked_docs = self.rank_documents(query, candidate_docs)
        
        # Định dạng kết quả
        results = []
        for rank, (doc_id, score) in enumerate(ranked_docs[:top_k]):
            doc = self.documents[doc_id]
            results.append({
                'rank': rank + 1,
                'doc_id': doc_id,
                'title': doc.title,
                'content': doc.content[:200] + "...",
                'url': doc.url,
                'author': doc.author,
                'score': float(score),
                'views': doc.views,
                'timestamp': doc.timestamp.isoformat() if isinstance(doc.timestamp, datetime) else str(doc.timestamp)
            })
        
        # Ghi nhận truy vấn
        self.query_log.append(query)
        
        return results
    
    def log_click(self, query_id: str, doc_id: str, position: int, dwell_time: float):
        """Ghi nhận nhấp chuột của người dùng"""
        click = ClickLog(
            query_id=query_id,
            doc_id=doc_id,
            position=position,
            dwell_time=dwell_time,
            timestamp=datetime.now(),
            user_id="user"
        )
        self.click_log.append(click)
        
        # Cập nhật thống kê tài liệu
        if doc_id in self.documents:
            doc = self.documents[doc_id]
            doc.clicks += 1
            doc.dwell_time = (doc.dwell_time * (doc.clicks - 1) + dwell_time) / doc.clicks
        
        # Cập nhật thống kê nhấp chuột
        key = (query_id, doc_id)
        if key not in self.click_stats:
            self.click_stats[key] = {'clicks': 0, 'dwell_time': 0}
        self.click_stats[key]['clicks'] += 1
        self.click_stats[key]['dwell_time'] += dwell_time
    
    def generate_training_pairs(self, n_queries: int = 100) -> List[PairwiseExample]:
        """Tạo các cặp huấn luyện tổng hợp"""
        pairs = []
        
        # Các truy vấn mẫu
        query_templates = [
            'machine learning', 'deep learning', 'neural networks',
            'python programming', 'data science', 'web development'
        ]
        
        for _ in range(n_queries):
            # Tạo truy vấn
            query_text = random.choice(query_templates)
            if random.random() > 0.5:
                query_text += f" {random.choice(['tutorial', 'guide', 'example'])}"
            
            query = Query(
                query_id=f"train_q_{len(pairs)}",
                text=query_text,
                user_id="trainer",
                timestamp=datetime.now()
            )
            
            # Lấy tài liệu ứng viên
            candidate_docs = self.initial_retrieval(query, top_k=20)
            
            if len(candidate_docs) < 2:
                continue
            
            # Tạo điểm liên quan dựa trên sự trùng lắp văn bản
            doc_relevances = []
            for doc_id in candidate_docs:
                doc = self.documents[doc_id]
                query_terms = set(query_text.lower().split())
                doc_terms = set(doc.title.lower().split() + doc.content.lower().split())
                overlap = len(query_terms & doc_terms)
                relevance = min(overlap / len(query_terms), 1.0)
                doc_relevances.append((doc_id, relevance))
            
            # Sắp xếp theo độ liên quan
            doc_relevances.sort(key=lambda x: x[1], reverse=True)
            
            # Tạo các cặp
            for i in range(min(5, len(doc_relevances) - 1)):
                for j in range(i + 1, min(i + 5, len(doc_relevances))):
                    doc_i_id, rel_i = doc_relevances[i]
                    doc_j_id, rel_j = doc_relevances[j]
                    
                    if abs(rel_i - rel_j) < 0.1:  # Bỏ qua độ liên quan tương tự
                        continue
                    
                    # Trích xuất đặc trưng
                    doc_i = self.documents[doc_i_id]
                    doc_j = self.documents[doc_j_id]
                    
                    features_i = self.feature_extractor.extract(
                        query, doc_i, self.corpus_stats, self.click_stats
                    )
                    features_j = self.feature_extractor.extract(
                        query, doc_j, self.corpus_stats, self.click_stats
                    )
                    
                    pair = PairwiseExample(
                        query_id=query.query_id,
                        doc_i_id=doc_i_id,
                        doc_j_id=doc_j_id,
                        features_i=features_i,
                        features_j=features_j,
                        label=1.0,  # doc_i ranks higher
                        weight=abs(rel_i - rel_j)  # Weight by relevance difference
                    )
                    pairs.append(pair)
        
        return pairs
    
    def train(self, epochs: int = 50, batch_size: int = 32):
        """Huấn luyện mô hình RankNet"""
        print("Đang tạo dữ liệu huấn luyện...")
        training_pairs = self.generate_training_pairs(n_queries=200)
        
        if not training_pairs:
            print("Không có cặp huấn luyện được tạo!")
            return
        
        # Chia tập huấn luyện/kiểm định
        n_train = int(len(training_pairs) * 0.8)
        train_pairs = training_pairs[:n_train]
        val_pairs = training_pairs[n_train:]
        
        print(f"Cặp huấn luyện: {len(train_pairs)}, Cặp kiểm định: {len(val_pairs)}")
        
        # Khởi tạo RankNet
        self.ranknet = RankNet(
            input_dim=self.feature_extractor.n_features,
            hidden_dims=[64, 32],
            learning_rate=0.001
        )
        
        # Huấn luyện
        trainer = RankNetTrainer(self.ranknet)
        history = trainer.train(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True
        )
        
        print("Huấn luyện hoàn tất!")
        return history
    
    def evaluate(self, test_queries: List[Query] = None) -> Dict[str, float]:
        """Đánh giá hiệu suất tìm kiếm"""
        if not test_queries:
            # Tạo các truy vấn kiểm thử
            test_queries = []
            for i in range(20):
                query = Query(
                    query_id=f"test_q_{i}",
                    text=random.choice(['machine learning', 'python', 'data science']),
                    user_id="tester",
                    timestamp=datetime.now()
                )
                
                # Tạo điểm liên quan ngẫu nhiên
                query.relevance_scores = {}
                for doc_id in random.sample(list(self.documents.keys()), 
                                           min(10, len(self.documents))):
                    query.relevance_scores[doc_id] = random.choice([0, 1, 2, 3, 4])
                
                test_queries.append(query)
        
        ndcg_scores = []
        for query in test_queries:
            if not query.relevance_scores:
                continue
            
            # Lấy kết quả tìm kiếm
            results = self.search(query.text)
            ranked_docs = [r['doc_id'] for r in results]
            
            # Tính NDCG
            ndcg = compute_ndcg(ranked_docs, query.relevance_scores, k=10)
            ndcg_scores.append(ndcg)
        
        return {
            'ndcg@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'num_queries': len(test_queries)
        }
    
    def save(self, directory: str):
        """Lưu trạng thái công cụ tìm kiếm"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Lưu tài liệu
        with open(f"{directory}/documents.pkl", 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Lưu thống kê kho dữ liệu
        with open(f"{directory}/corpus_stats.pkl", 'wb') as f:
            pickle.dump(dict(self.corpus_stats), f)
        
        # Lưu mô hình
        if self.ranknet:
            self.ranknet.save(f"{directory}/ranknet.pkl")
        
        print(f"Đã lưu công cụ tìm kiếm vào {directory}")
    
    def load(self, directory: str):
        """Tải trạng thái công cụ tìm kiếm"""
        # Tải tài liệu
        with open(f"{directory}/documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        # Xây dựng lại chỉ mục
        for doc in self.documents.values():
            self._update_index(doc)
        
        # Tải thống kê kho dữ liệu
        with open(f"{directory}/corpus_stats.pkl", 'rb') as f:
            self.corpus_stats = pickle.load(f)
        
        # Tải mô hình
        try:
            self.ranknet = RankNet(input_dim=self.feature_extractor.n_features)
            self.ranknet.load(f"{directory}/ranknet.pkl")
        except:
            print("Không tìm thấy mô hình đã lưu")
        
        print(f"Đã tải công cụ tìm kiếm từ {directory}")