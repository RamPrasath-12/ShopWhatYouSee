"""
Test script for Unified LLM (Local + External fallback)

Run from backend directory:
    python test_unified_llm.py

Tests both local Phi-3 and external Groq/Llama integration.
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.unified_llm import generate_filters


def test_color_change():
    """Test: User wants different color."""
    print("\n" + "="*60)
    print("TEST 1: Color Change")
    print("="*60)
    
    result = generate_filters(
        category="tshirt",
        attributes={"color": "blue", "pattern": "solid", "sleeve": "short"},
        scene="casual street",
        query="show me something similar but in red"
    )
    
    print(f"Source: {result.get('source', 'unknown')}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    print(f"Filters: {result.get('filters', {})}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    return result


def test_price_query():
    """Test: User wants budget option."""
    print("\n" + "="*60)
    print("TEST 2: Price/Budget Query")
    print("="*60)
    
    result = generate_filters(
        category="jacket",
        attributes={"color": "black", "pattern": "solid"},
        scene="formal office",
        query="I want something cheaper, under 2000 rupees"
    )
    
    print(f"Source: {result.get('source', 'unknown')}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    print(f"Filters: {result.get('filters', {})}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    return result


def test_pattern_change():
    """Test: User wants different pattern."""
    print("\n" + "="*60)
    print("TEST 3: Pattern Change")
    print("="*60)
    
    result = generate_filters(
        category="shirt",
        attributes={"color": "white", "pattern": "solid", "sleeve": "long"},
        scene="party",
        query="show me striped patterns instead"
    )
    
    print(f"Source: {result.get('source', 'unknown')}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    print(f"Filters: {result.get('filters', {})}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    return result


def test_iterative_query():
    """Test: Multi-turn conversation."""
    print("\n" + "="*60)
    print("TEST 4: Iterative/Multi-turn Query")
    print("="*60)
    
    result = generate_filters(
        category="tshirt",
        attributes={"color": "blue", "pattern": "solid", "sleeve": "short"},
        scene="casual",
        query="now show me with floral print",
        session_history=[
            {"query": "show me in red", "filters": {"color": "red"}},
            {"query": "make it long sleeve", "filters": {"sleeve": "long"}}
        ]
    )
    
    print(f"Source: {result.get('source', 'unknown')}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    print(f"Filters: {result.get('filters', {})}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    return result


def test_external_direct():
    """Test: Force external LLM."""
    print("\n" + "="*60)
    print("TEST 5: External LLM Direct (Groq)")
    print("="*60)
    
    result = generate_filters(
        category="pant",
        attributes={"color": "navy", "pattern": "solid"},
        scene="office",
        query="show me something in khaki color for casual friday",
        prefer_external=True
    )
    
    print(f"Source: {result.get('source', 'unknown')}")
    print(f"Reasoning: {result.get('reasoning', 'N/A')}")
    print(f"Filters: {result.get('filters', {})}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    return result


def main():
    print("="*60)
    print("UNIFIED LLM TEST SUITE")
    print("Testing Local (Phi-3) + External (Groq/Llama) Integration")
    print("="*60)
    
    # Run all tests
    results = []
    
    # Test 1: Color change
    results.append(("Color Change", test_color_change()))
    
    # Test 2: Price query  
    results.append(("Price Query", test_price_query()))
    
    # Test 3: Pattern change
    results.append(("Pattern Change", test_pattern_change()))
    
    # Test 4: Multi-turn
    results.append(("Iterative Query", test_iterative_query()))
    
    # Test 5: External direct
    results.append(("External Direct", test_external_direct()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, result in results:
        source = result.get('source', 'unknown')
        confidence = result.get('confidence', 0)
        filters = result.get('filters', {})
        status = "✅" if filters else "⚠️"
        print(f"{status} {name}: source={source}, confidence={confidence:.2f}, filters={filters}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
