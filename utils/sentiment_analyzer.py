"""
Sentiment Analyzer Utility
Simple wrapper for sentiment analysis on route descriptions
"""

from transformers import pipeline
import os

class SentimentAnalyzer:
    """Simple sentiment analyzer for route descriptions"""
    
    def __init__(self, model_path="utils/models/sentiment"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_path: Path to downloaded model
        """
        self.model_path = model_path
        self.sentiment_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load sentiment model"""
        if os.path.exists(self.model_path):
            print(f"📂 Loading model from: {self.model_path}")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_path,
                tokenizer=self.model_path
            )
            print("✅ Model loaded")
        else:
            print(f"⚠️  Model not found at: {self.model_path}")
            print("   Run: python download_sentiment_model.py")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
    
    def analyze(self, text):
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: {'label': 'POSITIVE'/'NEGATIVE', 'score': 0.95, 'sentiment_score': 0.95}
        """
        result = self.sentiment_pipeline(text)[0]
        
        # Convert to numerical score (-1 to 1)
        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        
        return {
            'label': result['label'],
            'confidence': result['score'],
            'sentiment_score': score  # -1 (negative) to 1 (positive)
        }
    
    def analyze_route(self, route_info):
        """
        Analyze sentiment for a route based on predicted noise
        
        Args:
            route_info: dict with 'street_name', 'neighborhood', 'predicted_noise'
            
        Returns:
            dict: sentiment analysis result
        """
        # Generate description based on noise
        noise = route_info['predicted_noise']
        street = route_info['street_name']
        neighborhood = route_info['neighborhood']
        
        if noise >= 90:
            description = f"Heavy traffic on {street} in {neighborhood}. Very loud and congested area with significant noise pollution."
        elif noise >= 80:
            description = f"Busy route through {street}, {neighborhood}. Moderate to high traffic noise expected."
        elif noise >= 70:
            description = f"Normal traffic conditions on {street} in {neighborhood}. Acceptable noise levels."
        else:
            description = f"Quiet route through {street}, {neighborhood}. Low traffic and peaceful environment."
        
        # Analyze sentiment
        sentiment = self.analyze(description)
        sentiment['description'] = description
        
        return sentiment
    
    def batch_analyze(self, texts):
        """
        Analyze multiple texts at once
        
        Args:
            texts: List of texts
            
        Returns:
            list: List of sentiment results
        """
        results = self.sentiment_pipeline(texts)
        
        return [
            {
                'label': r['label'],
                'confidence': r['score'],
                'sentiment_score': r['score'] if r['label'] == 'POSITIVE' else -r['score']
            }
            for r in results
        ]
