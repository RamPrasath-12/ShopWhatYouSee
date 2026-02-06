"""
Simple LRU Cache for LLM Responses
Caches responses to avoid re-computing identical queries
"""
import hashlib
import json
import time
from collections import OrderedDict
from typing import Dict, Any, Optional

class LLMCache:
    def __init__(self, max_size=100, ttl=3600):
        """
        Args:
            max_size: Maximum number of cached responses
            ttl: Time to live in seconds (default 1 hour)
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        
    def _generate_key(self, category: str, query: str, attributes: Dict[str, Any]) -> str:
        """Generate cache key from request parameters"""
        # Create deterministic hash from inputs
        data = {
            "category": category,
            "query": query.lower().strip(),
            "attributes": attributes
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def get(self, category: str, query: str, attributes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response if available and not expired"""
        key = self._generate_key(category, query, attributes)
        
        if key in self.cache:
            entry = self.cache[key]
            # Check if expired
            if time.time() - entry["timestamp"] < self.ttl:
                # Move to end (mark as recently used)
                self.cache.move_to_end(key)
                print(f"[Cache] ⚡ HIT for key {key[:8]}...")
                return entry["response"]
            else:
                # Expired, remove
                del self.cache[key]
                print(f"[Cache] ⏱️ EXPIRED for key {key[:8]}...")
        
        print(f"[Cache] ❌ MISS for key {key[:8]}...")
        return None
    
    def set(self, category: str, query: str, attributes: Dict[str, Any], response: Dict[str, Any]):
        """Cache a response"""
        key = self._generate_key(category, query, attributes)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest (first item)
        
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
        print(f"[Cache] ✅ STORED key {key[:8]}...")
    
    def clear(self):
        """Clear all cached responses"""
        self.cache.clear()
        print("[Cache] 🧹 Cleared")

# Global cache instance
_cache = None

def get_cache() -> LLMCache:
    """Get global cache instance"""
    global _cache
    if _cache is None:
        _cache = LLMCache()
    return _cache
