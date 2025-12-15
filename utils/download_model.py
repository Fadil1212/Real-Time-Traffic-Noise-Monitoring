#!/usr/bin/env python3
"""
Download Lightweight Sentiment Analysis Model
Saves to: models/sentiment/
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def download_sentiment_model():
    """
    Download a small, efficient sentiment model
    
    Model: distilbert-base-uncased-finetuned-sst-2-english
    - Size: ~255 MB
    - Speed: Fast
    - Accuracy: Good (91% on SST-2)
    - Perfect for production use
    """
    
    print("=" * 70)
    print("DOWNLOADING SENTIMENT ANALYSIS MODEL")
    print("=" * 70)
    print()
    
    # Model name (lightweight distilbert)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Save path
    save_path = "models/sentiment"
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Model: {model_name}")
    print(f"Save path: {save_path}")
    print(f"Size: ~255 MB")
    print()
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        print("Tokenizer saved")
        
        # Download model
        print("\nDownloading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.save_pretrained(save_path)
        print("Model saved")
        
        print()
        print("=" * 70)
        print("DOWNLOAD COMPLETE!")
        print("=" * 70)
        print()
        print(f"Model location: {os.path.abspath(save_path)}")
        print()
        print("📊 Model Details:")
        print(f"   Name: DistilBERT (distilled BERT)")
        print(f"   Parameters: 66M (small!)")
        print(f"   Accuracy: 91% on SST-2")
        print(f"   Speed: ~20ms per prediction")
        print(f"   Labels: POSITIVE, NEGATIVE")
        print()
        print("🔍 Files saved:")
        for file in os.listdir(save_path):
            size = os.path.getsize(os.path.join(save_path, file)) / (1024 * 1024)
            print(f"   - {file:<30} ({size:>6.1f} MB)")
        
        print()
        print("🎯 Next step: Use the model in your code!")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    # Download
    download_sentiment_model()