# dag.py
from nodes.inference_node import InferenceNode
from nodes.confidence_node import ConfidenceCheckNode
from nodes.fallback_node import FallbackNode
from logger import JSONLogger

class SimpleLangGraphDAG:
    """
    Wires the nodes together and provides a run(text) interface that returns
    the final label, confidence, and a list of step logs.
    """

    def __init__(self, model_dir="saved_model", threshold=0.70, fallback_strategy="ask_then_zero_shot", log_file="demo_logs.log"):
        self.inference = InferenceNode(model_dir=model_dir)
        self.confidence = ConfidenceCheckNode(threshold=threshold)
        self.fallback = FallbackNode(fallback_strategy=fallback_strategy)
        self.logger = JSONLogger(log_file)

    def run(self, text: str):
        # 1) Inference
        label, confidence, logits = self.inference.predict(text)
        self.logger.log("inference", {"input_text": text, "predicted_label": label, "confidence": confidence})
        print(f"[InferenceNode] Predicted label: {label} | Confidence: {confidence*100:.1f}%")

        # 2) Confidence check
        ok, message = self.confidence.check(label, confidence)
        self.logger.log("confidence_check", {"predicted_label": label, "confidence": confidence, "threshold": self.confidence.threshold, "result": ok, "message": message})
        print(f"[ConfidenceCheckNode] {message}")

        if ok:
            # Accept prediction
            self.logger.log("final_decision", {"final_label": label, "accepted": True, "reason": "confidence_ok"})
            print(f"Final Label: {label} (Accepted automatically)")
            return {"final_label": label, "confidence": confidence, "accepted": True, "reason": "confidence_ok"}
        else:
            # Trigger fallback
            self.logger.log("fallback_trigger", {"predicted_label": label, "confidence": confidence})
            print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
            final_label, reason = self.fallback.run(text, candidate_labels=list(self.inference.label_map.values()))
            # log fallback result
            self.logger.log("fallback_result", {"final_label": final_label, "reason": reason})
            self.logger.log("final_decision", {"final_label": final_label, "accepted": False, "reason": reason})
            print(f"Final Label: {final_label} ({'Corrected via fallback' if reason else 'Fallback result'})")
            return {"final_label": final_label, "confidence": confidence, "accepted": False, "reason": reason}

