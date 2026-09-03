import os
import sys

# Ensure the root directory is on the path so app.py and backend/ can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app