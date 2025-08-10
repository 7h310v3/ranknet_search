# Cấu trúc dự án RankNet
```
ranknet_search/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/              # Dataset files
│   └── models/           # Saved models
│
├── src/
│   ├── __init__.py
│   ├── models.py         # Document, Query, PairwiseExample classes
│   ├── features.py       # All feature extraction in one file
│   ├── ranknet.py        # RankNet model (both NumPy & PyTorch)
│   ├── engine.py         # Search engine + training logic
│   └── utils.py          # Metrics, helpers
│
├── api/
│   └── app.py            # FastAPI server
│
│
├── scripts/
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
│
└── config.yaml           # Configuration
```

## 📝 File chính:

### **src/models.py** (~150 lines)
```python
# Chứa tất cả data models
@dataclass
class Document: ...

@dataclass
class Query: ...

@dataclass
class PairwiseExample: ...
```

### **src/features.py**
```python
# Một class chứa tất cả feature extraction
class FeatureExtractor:
    def extract_text_features(): ...
    def extract_click_features(): ...
    def extract_quality_features(): ...
    def extract(): ...  # Main method
```

### **src/ranknet.py**
```python
# RankNet model và training
class RankNet:
    def __init__(): ...
    def forward(): ...
    def train_batch(): ...
    def predict(): ...

class RankNetTrainer:
    def train(): ...
    def evaluate(): ...
```

### **src/engine.py**
```python
# Search engine chính
class SearchEngine:
    def __init__(): ...
    def add_document(): ...
    def load_data(): ...  # Data loading
    def generate_training_data(): ...  # Data generation
    def train_ranknet(): ...
    def search(): ...
    def save(): ...
    def load(): ...
```

### **api/app.py**
```python
# FastAPI với tất cả endpoints
app = FastAPI()

# Inline Pydantic models
class SearchRequest(BaseModel): ...
class SearchResponse(BaseModel): ...

@app.post("/search")
@app.post("/feedback")
@app.post("/train")
# ... all endpoints in one file
```
