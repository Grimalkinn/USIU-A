
import os
import requests

def downloader():
    """Download the pose landmarker model"""
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # URL for the default pose landmarker model
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    
    model_path = "./models/pose_landmarker_lite.task"
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model already exists: {model_path} ({size_mb:.1f} MB)")
        return model_path
    
    print(f"Downloading pose landmarker model...")
    print(f"  From: {model_url}")
    print(f"  To: {model_path}")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\r  Progress: {progress:.1f}%", end='')
        
        print(f"\n✓ Model downloaded successfully: {model_path}")
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
        return model_path
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return None

if __name__ == "__main__":
    downloader()