"""
Setup script to install accelerate and update dependencies for proper meta tensor handling
"""

#!/bin/bash

# Upgrade transformers and related packages
pip install --upgrade transformers>=4.30.0
pip install --upgrade torch>=2.0.0
pip install --upgrade accelerate>=0.19.0

# Install other dependencies if not already installed


# Print versions for verification
echo "Checking installed versions:"
pip show torch | grep Version
pip show transformers | grep Version
pip show accelerate | grep Version

echo "Installation complete. You can now run the model with proper meta tensor handling."
echo "Example command:"
echo "python .\main.py --enhanced --embedding-model nomic-ai/nomic-embed-code --max-length 1024 --batch-size 16 --folder <path/to/your/docs> --reset --db-dir "../chroma_db""
echo "python .\main.py --enhanced --embedding-model nomic-ai/nomic-embed-code --max-length 1024 --batch-size 16 --folder <path/to/your/code> --db-dir "../chroma_db""