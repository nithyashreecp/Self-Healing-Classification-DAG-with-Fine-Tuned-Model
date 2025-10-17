# cli.py
import argparse
from nodes.dag import SimpleLangGraphDAG


def main():
    parser = argparse.ArgumentParser(description="ATG Self-Healing Classification CLI")
    parser.add_argument("--model_dir", type=str, default="saved_model", help="Directory with the saved model/tokenizer")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (0-1) to accept prediction")
    parser.add_argument("--fallback", type=str, default="ask_then_zero_shot", choices=["ask_user","zero_shot","ask_then_zero_shot"], help="Fallback strategy")
    args = parser.parse_args()

    dag = SimpleLangGraphDAG(model_dir=args.model_dir, threshold=args.threshold, fallback_strategy=args.fallback)

    print("=== ATG Self-Healing Classification CLI ===")
    print("Type a sentence to classify (type 'exit' to quit).")
    while True:
        text = input("\nInput: ").strip()
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            print("Exiting CLI.")
            break
        result = dag.run(text)
        # For CLI, show final result clearly
        print(f"[CLI] Result -> Label: {result['final_label']}, Confidence of model: {result['confidence']*100:.1f}%, Accepted: {result['accepted']}, Reason: {result['reason']}")

if __name__ == "__main__":
    main()



