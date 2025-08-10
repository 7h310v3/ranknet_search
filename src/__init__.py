"""
src/__init__.py - Package initialization
"""

# Import main components for easier access
from .engine import SearchEngine
from .features import FeatureExtractor
from .ranknet import RankNet, RankNetTrainer
from .model import Document, Query, PairwiseExample, ClickLog
