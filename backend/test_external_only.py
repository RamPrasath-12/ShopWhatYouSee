"""
Quick test for External LLM Only (Groq)
This skips the local Phi-3 model entirely.

Run: python test_external_only.py
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Verify API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("❌ GROQ_API_KEY not found in .env file!")
    print("Add: GROQ_API_KEY=your_key_here")
    exit(1)
else:
    print(f"✅ GROQ_API_KEY found: {api_key[:10]}...")

# Import and test external LLM directly
from models.external_llm import generate_filters_external

print("\n" + "="*60)
print("TEST 1: Color Change (External LLM)")
print("="*60)

result = generate_filters_external(
    category="tshirt",
    attributes={"color": "blue", "pattern": "solid", "sleeve": "short"},
    scene="casual street",
    query="show me something similar but in red"
)

print(f"Reasoning: {result.get('reasoning', 'N/A')}")
print(f"Filters: {result.get('filters', {})}")
print(f"Confidence: {result.get('confidence', 0):.2f}")
print(f"Source: {result.get('source', 'unknown')}")

print("\n" + "="*60)
print("TEST 2: Price Query (External LLM)")
print("="*60)

result = generate_filters_external(
    category="jacket",
    attributes={"color": "black"},
    scene="office",
    query="show me something budget friendly under 1500"
)

print(f"Reasoning: {result.get('reasoning', 'N/A')}")
print(f"Filters: {result.get('filters', {})}")
print(f"Confidence: {result.get('confidence', 0):.2f}")

print("\n" + "="*60)
print("TEST 3: Pattern + Multi-turn (External LLM)")
print("="*60)

result = generate_filters_external(
    category="shirt",
    attributes={"color": "white", "pattern": "solid"},
    scene="party",
    query="show me with stripes now",
    session_history=[
        {"query": "show me in blue", "filters": {"color": "blue"}}
    ]
)

print(f"Reasoning: {result.get('reasoning', 'N/A')}")
print(f"Filters: {result.get('filters', {})}")
print(f"Confidence: {result.get('confidence', 0):.2f}")

print("\n✅ External LLM (Groq) tests complete!")
print("The Phi-3 local model will work once its download completes.")
