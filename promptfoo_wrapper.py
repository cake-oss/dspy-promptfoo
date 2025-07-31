#!/usr/bin/env python3
"""
Wrapper script for Promptfoo to ensure proper environment setup
This script ensures that DSPy and other dependencies are available
"""

import sys
import os
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

# Now import the actual provider
from src.dspy_promptfoo.provider import call_api

# Re-export the call_api function
__all__ = ['call_api']
