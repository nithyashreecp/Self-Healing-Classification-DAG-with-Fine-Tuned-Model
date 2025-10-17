# nodes/inference_node.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, Dict

class InferenceNode:
    """
    Loads a fine-tuned classification model from saved_model/ and performs
    inference producing (label, confidence, raw_logits).
    """

    def __init__(self, model_dir: str = "saved_model", device: str = None, label_map: Dict[int, str] = None):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        # label_map maps model output indices to human labels
        self.label_map = label_map or {0: "NEGATIVE", 1: "POSITIVE"}

    def predict(self, text: str) -> Tuple[str, float, torch.Tensor]:
        """
        Returns (label_str, confidence_float, logits_tensor)
        confidence is the probability of the predicted class (softmax).
        """
        self.model.eval()
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape (1, num_labels)
            probs = F.softmax(logits, dim=-1).squeeze(0)  # shape (num_labels,)
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
            label_str = self.label_map.get(pred_idx, str(pred_idx))
        return label_str, confidence, logits.cpu()




