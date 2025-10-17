# nodes/confidence_node.py
from typing import Tuple

class ConfidenceCheckNode:
    """
    Given a label and confidence score, decide whether to accept or trigger fallback.
    """

    def __init__(self, threshold: float = 0.70):
        """
        threshold: float between 0 and 1 representing minimum acceptable confidence
        """
        self.threshold = threshold

    def check(self, label: str, confidence: float) -> Tuple[bool, str]:
        """
        Returns (is_confident, message)
        """
        if confidence >= self.threshold:
            return True, f"Confidence ({confidence*100:.1f}%) >= threshold ({self.threshold*100:.0f}%). Accepting prediction."
        else:
            return False, f"Confidence ({confidence*100:.1f}%) < threshold ({self.threshold*100:.0f}%). Triggering fallback."



