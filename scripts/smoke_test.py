"""Sanity check: load the model and answer one GSM8K-style question.

  python scripts/smoke_test.py

If it prints a worked solution ending in a number, the local stack works.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import load_model, sample          # noqa: E402
from eval.gsm8k import extract_answer             # noqa: E402

MODEL = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B"
Q = ("Natalia sold clips to 48 friends in April, and then sold half as many in May. "
     "How many clips did she sell altogether?")


def main():
    print(f"loading {MODEL} (first run downloads weights)...")
    model, tokenizer = load_model(MODEL)
    print("\nQ:", Q)
    s = sample(model, tokenizer, Q, temp=0.0, max_tokens=512)
    print("\nresponse:\n" + s.text)
    print(f"\nextracted: {extract_answer(s.text)} (expected 72)")


if __name__ == "__main__":
    main()
