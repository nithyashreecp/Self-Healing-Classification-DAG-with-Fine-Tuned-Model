# nodes/fallback_node.py
from typing import Tuple, Optional
import logging

# Optional: zero-shot fallback using a second model (requires transformers pipeline)
try:
    from transformers import pipeline
    ZERO_SHOT_AVAILABLE = True
except Exception:
    ZERO_SHOT_AVAILABLE = False

class FallbackNode:
    """
    Two fallback strategies:
      1) Ask the user for clarification (interactive)
      2) Use a backup zero-shot classifier (if installed and desired)
    """

    def __init__(self, fallback_strategy: str = "ask_user", zero_shot_model: str = "facebook/bart-large-mnli"):
        """
        fallback_strategy: "ask_user" or "zero_shot" or "ask_then_zero_shot"
        """
        self.fallback_strategy = fallback_strategy
        self.zero_shot_model = zero_shot_model
        self.zero_shot = None
        if fallback_strategy in ("zero_shot", "ask_then_zero_shot") and ZERO_SHOT_AVAILABLE:
            # candidate labels for sentiment: NEGATIVE, POSITIVE
            self.zero_shot = pipeline("zero-shot-classification", model=zero_shot_model)

    def run(self, text: str, candidate_labels=None) -> Tuple[str, str]:
        """
        Returns (final_label, reason)
        """
        candidate_labels = candidate_labels or ["NEGATIVE", "POSITIVE"]
        # Strategy: ask user first if configured
        if self.fallback_strategy in ("ask_user", "ask_then_zero_shot"):
            print("[FallbackNode] Could you clarify your intent? (e.g. Was the review negative, positive, neutral?)")
            user = input("User (type label or free text; type 'skip' to skip): ").strip()
            if user.lower() in ("negative", "neg", "n"):
                return "NEGATIVE", "Corrected by user clarification"
            if user.lower() in ("positive", "pos", "p"):
                return "POSITIVE", "Corrected by user clarification"
            if user.lower() == "skip":
                # continue to other strategy
                pass
            else:
                # If user typed free text, allow treating it as additional input: ask yes/no mapping
                if user:
                    print("[FallbackNode] Thanks — interpreting user clarification as new evidence. Using zero-shot (if available) or asking again.")
                    # If zero-shot available, use it
                    if self.zero_shot is not None:
                        zs = self.zero_shot(user, candidate_labels)
                        label = zs["labels"][0]
                        score = zs["scores"][0]
                        return label, f"User free-text clarified; zero-shot interpreted as {label} ({score:.2f})"
                    else:
                        # fallback to simple heuristic: check 'not' / 'no' words
                        low = user.lower()
                        if "not" in low or "n't" in low or "no" in low:
                            return "NEGATIVE", "Heuristic from user text"
                        # final fallback: treat as POSITIVE
                        return "POSITIVE", "Heuristic from user text (default positive)"
        # Zero-shot fallback if configured
        if self.fallback_strategy in ("zero_shot", "ask_then_zero_shot") and self.zero_shot is not None:
            zs = self.zero_shot(text, candidate_labels)
            label = zs["labels"][0]
            score = zs["scores"][0]
            return label, f"Zero-shot fallback: {label} ({score:.2f})"
        # Last-resort: ask the user to type final label explicitly
        print("[FallbackNode] Please type the correct label now (NEGATIVE / POSITIVE):")
        user_label = input("Final label: ").strip().upper()
        if user_label in ("NEGATIVE", "POSITIVE"):
            return user_label, "Explicit user-provided final label"
        # default
        return "NEGATIVE", "Default fallback - chose NEGATIVE"





