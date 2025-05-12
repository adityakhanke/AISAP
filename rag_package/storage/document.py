"""
Document data classes for storing text chunks and metadata.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Document:
    """Represents a chunk of text from a file."""
    content: str
    metadata: Dict[str, Any]