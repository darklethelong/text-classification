"""
Run API Server Script
--------------------
This script runs the FastAPI server with the LSTM-CNN RoBERTa model.
"""

import os
import sys
import subprocess

def main():
    """Run the FastAPI server."""
    # Add src and api directories to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, 'src')
    api_path = os.path.join(project_root, 'api')
    
    sys.path.append(project_root)
    sys.path.append(src_path)
    sys.path.append(api_path)
    
    # Set environment variables
    os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{src_path}{os.pathsep}{api_path}"
    
    # Run the API server
    print("Starting the Complaint Detection API server with RoBERTa model...")
    os.chdir(api_path)
    subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    main() 