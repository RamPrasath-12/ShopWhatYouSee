"""
Quick script to add rating endpoints to app.py
Run this to inject the endpoints before if __main__
"""

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with if __main__
insert_index = None
for i, line in enumerate(lines):
    if '__main__' in line and 'if' in line:
        insert_index = i
        break

if insert_index is None:
    print("ERROR: Could not find if __main__ line")
    exit(1)

# Check if endpoints already exist
has_rating = any('/rating' in line for line in lines)
has_insights = any('/insights' in line for line in lines)

if has_rating and has_insights:
    print("✅ Endpoints already exist!")
    exit(0)

# Prepare the endpoints code
new_code = """
# --------------------------------------------------
# RATING COLLECTION ENDPOINT
# --------------------------------------------------
@app.route("/rating", methods=["POST"])
def rating_route():
    \"\"\"Collect user ratings for ML/UX improvement\"\"\"
    from rating_system import save_rating
    
    data = request.get_json() or {}
    
    # Validate rating
    rating = data.get("rating")
    if not rating or not isinstance(rating, int) or not (1 <= rating <= 5):
        return jsonify({"error": "Rating must be 1-5"}), 400
    
    success = save_rating(data)
    
    if success:
        return jsonify({"message": "Rating saved", "rating": rating})
    else:
        return jsonify({"error": "Failed to save rating"}), 500

# --------------------------------------------------
# INSIGHTS ENDPOINT FOR STAKEHOLDERS
# -------------------------------------------------- 
@app.route("/insights", methods=["GET"])
def insights_route():
    \"\"\"Get e-commerce insights based on user ratings\"\"\"
    from rating_system import get_insights
    
    insights = get_insights()
    return jsonify(insights)


"""

# Insert before if __main__
lines.insert(insert_index, new_code)

# Write back
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Successfully added rating & insights endpoints to app.py!")
