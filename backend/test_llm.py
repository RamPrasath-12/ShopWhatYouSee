# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model = "google/flan-t5-base"
# tok = AutoTokenizer.from_pretrained(model)
# m = AutoModelForSeq2SeqLM.from_pretrained(model)

# prompt = "Return JSON: {\"hello\": \"world\"}"
# inputs = tok(prompt, return_tensors="pt")
# out = m.generate(**inputs)
# print(tok.decode(out[0], skip_special_tokens=True))


"""
Test script for Local LLM filter generation
Run from backend directory:
    python -m tests.test_llm
"""

from models.local_llm import generate_filters


def run_test():
    result = generate_filters(
        category="tshirt",
        attributes={
            "color": "blue",
            "pattern": "solid"
        },
        scene="casual",
        query="show me in red"
    )

    print("\nGenerated Filters:")
    print(result)


if __name__ == "__main__":
    run_test()
