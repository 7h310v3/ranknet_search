"""
src/ranknet.py - Mạng nơ-ron RankNet cho học xếp hạng
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import pickle
from .model import PairwiseExample


class RankNet:
    """Hiện thực RankNet sử dụng NumPy"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 learning_rate: float = 0.001, sigma: float = 1.0):
        """
        Khởi tạo RankNet
        
        Tham số:
            input_dim: Kích thước đặc trưng đầu vào
            hidden_dims: Kích thước các lớp ẩn
            learning_rate: Tốc độ học
            sigma: Tham số tỉ lệ sigmoid
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.sigma = sigma
        
        # Khởi tạo trọng số
        self._initialize_weights()
        
        # Lịch sử huấn luyện
        self.loss_history = []
        
    def _initialize_weights(self):
        """Khởi tạo trọng số mạng sử dụng phương pháp khởi tạo Xavier"""
        self.weights = []
        self.biases = []
        
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            w = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / (prev_dim + hidden_dim))
            b = np.zeros((1, hidden_dim))
            self.weights.append(w)
            self.biases.append(b)
            prev_dim = hidden_dim
        
        # Lớp đầu ra
        w = np.random.randn(prev_dim, 1) * np.sqrt(2.0 / (prev_dim + 1))
        b = np.zeros((1, 1))
        self.weights.append(w)
        self.biases.append(b)
        
        # Trạng thái tối ưu hóa Adam
        self._init_adam()
    
    def _init_adam(self):
        """Khởi tạo tham số cho tối ưu hóa Adam"""
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.t = 0
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Lan truyền xuôi qua mạng
        
        Tham số:
            x: Đặc trưng đầu vào
            
        Trả về:
            output: Đầu ra của mạng
            cache: Giá trị đã lưu trữ cho lan truyền ngược
        """
        cache = {'activations': [x], 'pre_activations': []}
        
        # Các lớp ẩn với hàm kích hoạt ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(cache['activations'][-1], self.weights[i]) + self.biases[i]
            cache['pre_activations'].append(z)
            a = np.maximum(0, z)  # ReLU
            cache['activations'].append(a)
        
        # Lớp đầu ra (không có hàm kích hoạt)
        z = np.dot(cache['activations'][-1], self.weights[-1]) + self.biases[-1]
        cache['pre_activations'].append(z)
        cache['activations'].append(z)
        
        return z, cache
    
    def compute_gradients(self, xi: np.ndarray, xj: np.ndarray, 
                         label: float) -> Tuple[Dict, float]:
        """
        Tính gradient cho một cặp tài liệu
        
        Tham số:
            xi: Đặc trưng của tài liệu i (nên xếp hạng cao hơn)
            xj: Đặc trưng của tài liệu j (nên xếp hạng thấp hơn)
            label: 1.0 nếu xi > xj, 0.0 trong trường hợp ngược lại
            
        Trả về:
            gradients: Gradient của trọng số và độ lệch
            loss: Giá trị hàm mất mát
        """
        # Lan truyền xuôi cho cả hai tài liệu
        si, cache_i = self.forward(xi)
        sj, cache_j = self.forward(xj)
        
        # Tính xác suất xi xếp hạng cao hơn xj
        s_diff = (si - sj) / self.sigma
        p_ij = 1.0 / (1.0 + np.exp(-s_diff))
        
        # Hàm mất mát cross-entropy
        eps = 1e-10
        loss = -label * np.log(p_ij + eps) - (1 - label) * np.log(1 - p_ij + eps)
        
        # Gradient của hàm mất mát theo hiệu điểm số
        grad_s = (p_ij - label) / self.sigma
        
        # Lan truyền ngược qua cả hai mạng
        grads_i = self._backprop(cache_i, grad_s)
        grads_j = self._backprop(cache_j, -grad_s)
        
        # Kết hợp gradient (trọng số được chia sẻ)
        gradients = {
            'weights': [grads_i['weights'][i] + grads_j['weights'][i] 
                       for i in range(len(self.weights))],
            'biases': [grads_i['biases'][i] + grads_j['biases'][i] 
                      for i in range(len(self.biases))]
        }
        
        return gradients, float(loss)
    
    def _backprop(self, cache: Dict, grad_output: np.ndarray) -> Dict:
        """
        Lan truyền ngược cho một tài liệu
        
        Tham số:
            cache: Giá trị đã lưu từ lan truyền xuôi
            grad_output: Gradient từ hàm mất mát
            
        Trả về:
            gradients: Gradient của trọng số và độ lệch
        """
        grad_weights = []
        grad_biases = []
        
        delta = grad_output
        
        # Lan truyền ngược qua các lớp
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient theo trọng số và độ lệch
            grad_w = np.dot(cache['activations'][i].T, delta)
            grad_b = np.sum(delta, axis=0, keepdims=True)
            
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)
            
            if i > 0:
                # Lan truyền ngược đến lớp trước
                delta = np.dot(delta, self.weights[i].T)
                # Gradient của ReLU
                delta = delta * (cache['pre_activations'][i-1] > 0)
        
        return {'weights': grad_weights, 'biases': grad_biases}
    
    def update_weights(self, gradients: Dict, batch_size: int):
        """
        Cập nhật trọng số sử dụng tối ưu hóa Adam
        
        Tham số:
            gradients: Gradient tích lũy
            batch_size: Kích thước batch để tính trung bình
        """
        self.t += 1
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        for i in range(len(self.weights)):
            # Trung bình gradient
            grad_w = gradients['weights'][i] / batch_size
            grad_b = gradients['biases'][i] / batch_size
            
            # Cập nhật Adam cho trọng số
            self.m_weights[i] = beta1 * self.m_weights[i] + (1 - beta1) * grad_w
            self.v_weights[i] = beta2 * self.v_weights[i] + (1 - beta2) * grad_w**2
            
            m_hat_w = self.m_weights[i] / (1 - beta1**self.t)
            v_hat_w = self.v_weights[i] / (1 - beta2**self.t)
            
            self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
            
            # Cập nhật Adam cho độ lệch
            self.m_biases[i] = beta1 * self.m_biases[i] + (1 - beta1) * grad_b
            self.v_biases[i] = beta2 * self.v_biases[i] + (1 - beta2) * grad_b**2
            
            m_hat_b = self.m_biases[i] / (1 - beta1**self.t)
            v_hat_b = self.v_biases[i] / (1 - beta2**self.t)
            
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
    
    def train_batch(self, pairs: List[PairwiseExample]) -> float:
        """
        Huấn luyện trên một batch các cặp tài liệu
        
        Tham số:
            pairs: Danh sách các cặp huấn luyện
            
        Trả về:
            avg_loss: Giá trị mất mát trung bình cho batch
        """
        # Tích lũy gradient
        acc_gradients = {
            'weights': [np.zeros_like(w) for w in self.weights],
            'biases': [np.zeros_like(b) for b in self.biases]
        }
        total_loss = 0.0
        
        for pair in pairs:
            # Đảm bảo đúng kích thước
            xi = pair.features_i.reshape(1, -1)
            xj = pair.features_j.reshape(1, -1)
            
            # Tính gradient
            grads, loss = self.compute_gradients(xi, xj, pair.label)
            
            # Tích lũy
            for i in range(len(self.weights)):
                acc_gradients['weights'][i] += grads['weights'][i] * pair.weight
                acc_gradients['biases'][i] += grads['biases'][i] * pair.weight
            
            total_loss += loss
        
        # Cập nhật trọng số
        self.update_weights(acc_gradients, len(pairs))
        
        avg_loss = total_loss / len(pairs)
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Dự đoán điểm liên quan
        
        Tham số:
            x: Mảng đặc trưng (n_samples, n_features)
            
        Trả về:
            scores: Điểm liên quan
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        scores, _ = self.forward(x)
        return scores.squeeze()
    
    def rank_documents(self, features_list: List[np.ndarray]) -> List[int]:
        """
        Xếp hạng tài liệu theo mức độ liên quan
        
        Tham số:
            features_list: Danh sách các vector đặc trưng
            
        Trả về:
            ranked_indices: Chỉ số tài liệu đã sắp xếp theo độ liên quan
        """
        features_array = np.array(features_list)
        scores = self.predict(features_array)
        ranked_indices = np.argsort(scores)[::-1]
        return ranked_indices.tolist()
    
    def save(self, filepath: str):
        """Lưu mô hình vào file"""
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'hidden_dims': self.hidden_dims,
            'input_dim': self.input_dim,
            'learning_rate': self.learning_rate,
            'sigma': self.sigma,
            'loss_history': self.loss_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """Tải mô hình từ file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.hidden_dims = model_data['hidden_dims']
        self.input_dim = model_data['input_dim']
        self.learning_rate = model_data['learning_rate']
        self.sigma = model_data['sigma']
        self.loss_history = model_data.get('loss_history', [])
        
        # Khởi tạo lại trạng thái Adam
        self._init_adam()


class RankNetTrainer:
    """Huấn luyện mô hình RankNet"""
    
    def __init__(self, ranknet: RankNet):
        self.ranknet = ranknet
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_pairs: List[PairwiseExample], 
             val_pairs: List[PairwiseExample] = None,
             epochs: int = 50, batch_size: int = 32,
             verbose: bool = True) -> Dict:
        """
        Huấn luyện mô hình RankNet
        
        Tham số:
            train_pairs: Cặp dữ liệu huấn luyện
            val_pairs: Cặp dữ liệu kiểm định
            epochs: Số epoch
            batch_size: Kích thước batch
            verbose: In tiến trình
            
        Trả về:
            history: Lịch sử huấn luyện
        """
        n_batches = len(train_pairs) // batch_size
        
        for epoch in range(epochs):
            # Xáo trộn dữ liệu huấn luyện
            np.random.shuffle(train_pairs)
            
            # Huấn luyện một epoch
            epoch_loss = 0.0
            for i in range(0, len(train_pairs), batch_size):
                batch = train_pairs[i:i+batch_size]
                if batch:
                    loss = self.ranknet.train_batch(batch)
                    epoch_loss += loss
            
            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_train_loss)
            
            # Kiểm định
            if val_pairs:
                val_loss = self.evaluate(val_pairs)
                self.val_losses.append(val_loss)
            
            # In tiến trình
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}"
                if val_pairs:
                    msg += f" - Val Loss: {val_loss:.4f}"
                print(msg)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def evaluate(self, pairs: List[PairwiseExample]) -> float:
        """Đánh giá trên các cặp dữ liệu kiểm định"""
        total_loss = 0.0
        
        for pair in pairs:
            xi = pair.features_i.reshape(1, -1)
            xj = pair.features_j.reshape(1, -1)
            _, loss = self.ranknet.compute_gradients(xi, xj, pair.label)
            total_loss += loss
        
        return total_loss / len(pairs) if pairs else 0.0