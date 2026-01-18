"""
Text utilities for cleaning review text and extracting topic attributes.

Provides functions for HTML unescaping, text normalization, and
keyword-based topic detection.
"""

import html
import re
from typing import Dict, List, Optional

# Pre-compiled regex patterns for efficiency
WHITESPACE_PATTERN = re.compile(r'\s+')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')


def clean_text(text: Optional[str]) -> str:
    """
    Clean review text by unescaping HTML entities and normalizing whitespace.

    Parameters
    ----------
    text : str or None
        Raw review text

    Returns
    -------
    str
        Cleaned text
    """
    if text is None or not isinstance(text, str):
        return ""

    # Unescape HTML entities (e.g., &#34; -> ", &amp; -> &)
    cleaned = html.unescape(text)

    # Remove HTML tags if any
    cleaned = HTML_TAG_PATTERN.sub(' ', cleaned)

    # Remove URLs
    cleaned = URL_PATTERN.sub(' ', cleaned)

    # Normalize whitespace (collapse multiple spaces, strip leading/trailing)
    cleaned = WHITESPACE_PATTERN.sub(' ', cleaned).strip()

    return cleaned


def get_text_length(text: Optional[str], cleaned: bool = True) -> int:
    """
    Get length of review text (character count).

    Parameters
    ----------
    text : str or None
        Review text
    cleaned : bool
        Whether to clean text first

    Returns
    -------
    int
        Character count
    """
    if text is None or not isinstance(text, str):
        return 0

    if cleaned:
        text = clean_text(text)

    return len(text)


def build_topic_regex(keywords: List[str]) -> re.Pattern:
    """
    Build a compiled regex pattern for topic keyword matching.

    Uses word boundaries for more accurate matching.

    Parameters
    ----------
    keywords : list
        List of keywords/phrases to match

    Returns
    -------
    re.Pattern
        Compiled regex pattern (case-insensitive)
    """
    # Sort by length (longest first) to handle overlapping keywords
    sorted_keywords = sorted(keywords, key=len, reverse=True)

    # Escape special regex characters and add word boundaries
    patterns = []
    for kw in sorted_keywords:
        escaped = re.escape(kw)
        # Use word boundaries for single words, looser matching for phrases
        if ' ' in kw:
            patterns.append(escaped)
        else:
            patterns.append(r'\b' + escaped + r'\b')

    pattern = '|'.join(patterns)
    return re.compile(pattern, re.IGNORECASE)


class TopicExtractor:
    """
    Extracts topic/attribute mentions from review text.

    Compiles regex patterns once for efficient repeated matching.

    Parameters
    ----------
    topic_keywords : dict
        Mapping of topic names to keyword lists
        Example: {"ValueShare": ["price", "cheap", "expensive"]}
    """

    def __init__(self, topic_keywords: Dict[str, List[str]]):
        self.topic_keywords = topic_keywords
        self.topic_patterns: Dict[str, re.Pattern] = {}

        # Pre-compile all patterns
        for topic_name, keywords in topic_keywords.items():
            self.topic_patterns[topic_name] = build_topic_regex(keywords)

        print(f"TopicExtractor initialized with {len(self.topic_patterns)} topics:")
        for name, keywords in topic_keywords.items():
            print(f"  {name}: {len(keywords)} keywords")

    def extract_topic_flags(self, text: str) -> Dict[str, int]:
        """
        Extract binary flags for each topic from text.

        Parameters
        ----------
        text : str
            Review text (cleaned or raw)

        Returns
        -------
        dict
            Mapping of topic names to binary flags (0 or 1)
        """
        # Clean text first
        cleaned = clean_text(text)

        flags = {}
        for topic_name, pattern in self.topic_patterns.items():
            flags[topic_name] = 1 if pattern.search(cleaned) else 0

        return flags

    def extract_topic_counts(self, text: str) -> Dict[str, int]:
        """
        Count number of matches for each topic in text.

        Parameters
        ----------
        text : str
            Review text

        Returns
        -------
        dict
            Mapping of topic names to match counts
        """
        cleaned = clean_text(text)

        counts = {}
        for topic_name, pattern in self.topic_patterns.items():
            matches = pattern.findall(cleaned)
            counts[topic_name] = len(matches)

        return counts

    def get_topic_names(self) -> List[str]:
        """Get list of topic names."""
        return list(self.topic_patterns.keys())


def extract_review_features(
    record: Dict,
    topic_extractor: TopicExtractor
) -> Dict:
    """
    Extract all features from a single review record.

    Parameters
    ----------
    record : dict
        Raw review record from JSONL
    topic_extractor : TopicExtractor
        Initialized topic extractor

    Returns
    -------
    dict
        Processed review with additional features
    """
    # Get text fields
    text = record.get("text", "") or ""
    title = record.get("title", "") or ""

    # Combine text and title for topic extraction
    full_text = f"{title} {text}"

    # Clean text and compute length
    cleaned_text = clean_text(full_text)
    text_len = len(cleaned_text)

    # Extract topic flags
    topic_flags = topic_extractor.extract_topic_flags(full_text)

    # Check for images
    images = record.get("images", []) or []
    has_image = 1 if len(images) > 0 else 0

    # Build processed record
    processed = {
        "parent_asin": record.get("parent_asin"),
        "asin": record.get("asin"),
        "user_id": record.get("user_id"),
        "timestamp": record.get("timestamp"),
        "rating": record.get("rating"),
        "helpful_vote": record.get("helpful_vote", 0) or 0,
        "verified_purchase": 1 if record.get("verified_purchase") else 0,
        "text_len": text_len,
        "has_image": has_image,
    }

    # Add topic flags
    processed.update(topic_flags)

    return processed


def validate_share(value: float, name: str = "share") -> float:
    """
    Validate that a share value is in [0, 1].

    Parameters
    ----------
    value : float
        Share value to validate
    name : str
        Name for error message

    Returns
    -------
    float
        Validated value (clipped to [0, 1] if out of range)
    """
    if value < 0:
        print(f"Warning: {name} = {value:.4f} < 0, clipping to 0")
        return 0.0
    if value > 1:
        print(f"Warning: {name} = {value:.4f} > 1, clipping to 1")
        return 1.0
    return value
