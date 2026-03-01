"""
Quick test for the local GGUF LLM
"""
import sys
sys.path.insert(0, '.')

from models.local_llm_efficient import LocalLLMEfficient

print("=" * 50)
print("Testing Local GGUF LLM (Your Fine-tuned Phi-3)")
print("=" * 50)

llm = LocalLLMEfficient()

print("\n1. Loading model...")
if not llm.load():
    print("❌ Failed to load model!")
    print("Make sure phi3-finetuned-q4.gguf is in backend/data/llm/")
    exit(1)

print("✅ Model loaded!")

print("\n2. Testing filter generation...")
result = llm.generate_filters(
    category="tshirt",
    attributes={"color": "blue", "pattern": "solid", "sleeve": "short"},
    scene="casual street",
    query="I want this in red instead",
    session_history=[]
)

print(f"\nResult:")
print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
print(f"  Filters: {result.get('filters', {})}")
print(f"  Confidence: {result.get('confidence', 0)}")

print("\n" + "=" * 50)
print("✅ Local LLM Test Complete!")
print("=" * 50)
