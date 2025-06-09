# run_app.py - Use this to start your Streamlit app
import os
import sys
import subprocess

# Set environment variables before importing anything
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fix torch issue before any imports
try:
    import torch
    # Patch the problematic torch.classes attribute
    if hasattr(torch, '_classes'):
        class MockPath:
            def __init__(self):
                self._path = []
        
        if not hasattr(torch._classes, '__path__'):
            torch._classes.__path__ = MockPath()
except ImportError:
    pass

# Now run streamlit
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app.py",  # Replace with your actual app filename
        "--server.fileWatcherType", "none",
        "--server.headless", "true"
    ])