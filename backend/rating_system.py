"""
Rating storage and insights generation for e-commerce stakeholders
Stores user ratings in PostgreSQL for production robustness
"""
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from typing import Dict, List, Any
import os

# PostgreSQL Connection Config (Matched with product_retrieval)
DB_CONFIG = {
    "host": "localhost",
    "database": "shopwhatyousee",
    "user": "postgres",
    "password": "postgres123@"
}

def get_db():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

def init_db():
    """Initialize ratings table in Postgres"""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id SERIAL PRIMARY KEY,
                product_id TEXT,
                rating INTEGER,
                query TEXT,
                filters TEXT,
                llm_source TEXT,
                timestamp TIMESTAMP,
                session_id TEXT,
                iteration_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()
        print("[Ratings] Database initialized (PostgreSQL)")
    except Exception as e:
        print(f"[Ratings] DB Init Error: {e}")

def save_rating(data: Dict[str, Any]) -> bool:
    """Save a rating to database"""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ratings (product_id, rating, query, filters, llm_source, timestamp, session_id, iteration_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data.get("product_id"),
            data.get("rating"),
            data.get("query"),
            json.dumps(data.get("filters", {})),
            data.get("llm_source", "unknown"),
            datetime.now(),
            data.get("session_id"),
            data.get("iteration_count", 0)
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[Ratings] Error saving: {e}")
        return False

def get_insights() -> Dict[str, Any]:
    """Generate insights for e-commerce stakeholders"""
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Total ratings
        cur.execute("SELECT COUNT(*) as total FROM ratings")
        total = cur.fetchone()["total"]
        
        # Average rating
        cur.execute("SELECT AVG(rating) as avg_rating FROM ratings")
        res = cur.fetchone()["avg_rating"]
        avg_rating = float(res) if res else 0.0
        
        # Rating distribution
        cur.execute("SELECT rating, COUNT(*) as count FROM ratings GROUP BY rating ORDER BY rating")
        distribution = {row["rating"]: row["count"] for row in cur.fetchall()}
        
        # Low-rated searches (potential product gaps)
        cur.execute("""
            SELECT query, AVG(rating) as avg_rating, COUNT(*) as count
            FROM ratings
            GROUP BY query
            HAVING AVG(rating) < 3
            ORDER BY count DESC
            LIMIT 5
        """)
        low_rated = [dict(row) for row in cur.fetchall()]
        
        # High refinement queries (users struggling)
        cur.execute("""
            SELECT query, AVG(iteration_count) as avg_iterations, COUNT(*) as count
            FROM ratings
            WHERE iteration_count > 1
            GROUP BY query
            ORDER BY avg_iterations DESC
            LIMIT 5
        """)
        high_refinement = [dict(row) for row in cur.fetchall()]
        
        # LLM performance comparison
        cur.execute("""
            SELECT llm_source, AVG(rating) as avg_rating, COUNT(*) as count
            FROM ratings
            GROUP BY llm_source
        """)
        llm_performance = {row["llm_source"]: {"avg_rating": float(row["avg_rating"]), "count": row["count"]} 
                          for row in cur.fetchall()}
        
        conn.close()
        
        return {
            "total_ratings": total,
            "average_rating": round(avg_rating, 2),
            "distribution": distribution,
            "insights": {
                "product_gaps": low_rated,
                "user_struggles": high_refinement,
                "llm_performance": llm_performance
            },
            "recommendations": generate_recommendations(avg_rating, low_rated, high_refinement, llm_performance)
        }
    except Exception as e:
        print(f"[Insights] Error: {e}")
        return {"error": str(e), "total_ratings": 0, "average_rating": 0, "insights": {}}

def generate_recommendations(avg_rating, low_rated, high_refinement, llm_performance):
    """Generate actionable recommendations for stakeholders"""
    recommendations = []
    
    if avg_rating < 3.5:
        recommendations.append({
            "priority": "HIGH",
            "category": "Product Quality",
            "message": f"Overall rating ({avg_rating:.1f}/5) is below target. Improve product catalog relevance."
        })
    
    if low_rated:
        top_query = low_rated[0]["query"]
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Product Gaps",
            "message": f"Query '{top_query}' has low ratings. Consider adding more products matching this demand."
        })
    
    if high_refinement:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "Search UX",
            "message": "Users are refining searches multiple times. Improve initial filter accuracy or add suggestions."
        })
    
    # Compare Phi-3 vs Groq
    if "phi3_finetuned" in llm_performance and "groq" in llm_performance:
        phi3_rating = llm_performance["phi3_finetuned"]["avg_rating"]
        groq_rating = llm_performance["groq"]["avg_rating"]
        
        if phi3_rating > groq_rating + 0.3:
            recommendations.append({
                "priority": "LOW",
                "category": "ML Performance",
                "message": f"Fine-tuned Phi-3 outperforms Groq ({phi3_rating:.2f} vs {groq_rating:.2f}). Good contribution!"
            })
    
    return recommendations

def get_latest_rating():
    """Get the most recent rating for analysis"""
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM ratings ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        conn.close()
        
        if row:
            d = dict(row)
            # Parse filters if string
            if isinstance(d.get("filters"), str):
                try:
                    d["filters"] = json.loads(d["filters"])
                except:
                    d["filters"] = {}
            return d
        return None
    except Exception as e:
        print(f"[Ratings] Error fetching latest: {e}")
        return None

# Initialize on import
init_db()
