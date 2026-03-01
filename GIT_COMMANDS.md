# ============================================================
# GIT COMMANDS TO PUSH PROJECT TO NEW BRANCH
# ============================================================
# Run these commands in: D:\Final_Year_Project\ShopWhatYouSee

# 1. Check current status
git status

# 2. Create and switch to new branch (choose a meaningful name)
git checkout -b feature/llm-integration

# 3. Add all files (respecting .gitignore)
git add .

# 4. Check what will be committed (verify no large files)
git status

# 5. Commit with a descriptive message
git commit -m "Add LLM integration with local Phi-3 and Groq fallback

- Added fine-tuned Phi-3 GGUF model support for local inference
- Integrated Groq API as fallback for low-confidence queries
- Created unified LLM module with smart confidence-based fallback
- Added AGMAN model for product embeddings (87.4% Recall@5)
- Multi-task learning for discriminative embeddings"

# 6. Push to remote (first time for new branch)
git push -u origin feature/llm-integration

# ============================================================
# WHAT'S EXCLUDED (via .gitignore):
# ============================================================
# ✓ venv/                    - Virtual environment
# ✓ *.pth, *.gguf           - Model weights (too large)
# ✓ data/llm/               - LLM model files
# ✓ data/agman/             - AGMAN model files
# ✓ .env                    - API keys (NEVER upload!)
# ✓ __pycache__/            - Python cache
# ✓ *.safetensors, *.bin   - Model files

# ============================================================
# WHAT'S INCLUDED:
# ============================================================
# ✓ backend/models/*.py     - Your Python code
# ✓ frontend/               - React/JS code
# ✓ requirements.txt        - Dependencies
# ✓ README.md               - Documentation
# ✓ .gitignore              - Git configuration

# ============================================================
# IF YOU NEED TO SHARE MODEL FILES:
# ============================================================
# Use Google Drive, Hugging Face Hub, or Git LFS
# GitHub has a 100MB file size limit!
# 
# For Hugging Face:
#   pip install huggingface_hub
#   huggingface-cli login
#   huggingface-cli upload your-username/shop-what-you-see ./models
