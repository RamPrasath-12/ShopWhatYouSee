# ShopWhatYouSee - Project Setup Guide

Welcome to the ShopWhatYouSee repository! This guide will walk you through setting up the project from scratch on your local machine so you can develop and run the application.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- **Git**: To clone the repository.
- **Node.js**: (v16.x or higher) and npm (for the frontend).
- **Python**: (v3.9 or higher) and pip (for the backend).
- **PostgreSQL**: (Optional, if running a local DB instead of Supabase).

## 1. Clone the Repository

First, clone the project to your local machine:

```bash
git clone https://github.com/RamPrasath-12/ShopWhatYouSee.git
cd ShopWhatYouSee
```

---

## 2. Backend Setup

The backend is built with Python (Flask, PyTorch, Ultralytics YOLO). 

### Step 2.1: Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

**Windows:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
```

### Step 2.2: Install Dependencies
With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### Step 2.3: Environment Variables
Create a `.env` file in the `backend/` directory. You can use `.env.example` as a template:

```bash
cp .env.example .env
```

Open `.env` and fill in the required api keys:
- `GEMINI_API_KEY`: For the unified LLM reasoning.
- `GROQ_API_KEY`: For LLM filtering logic.
- `DATABASE_URL`: Ensure this points to the shared Supabase Postgres instance.
- `DASHBOARD_PASSWORD`: Password for the admin dashboard (e.g., swys2026).

*(Note: The `DATABASE_URL` is pre-configured in the example to connect to the cloud Supabase instance, meaning you don't need to run a local database!)*

### Step 2.4: Run the Backend server
Start the Flask API server:

```bash
python app.py
```
The backend should now be running on `http://127.0.0.1:5000`.

---

## 3. Frontend Setup

The frontend is built with React and Vite.

### Step 3.1: Install Dependencies
Open a **new terminal window**, navigate to the frontend directory, and install the npm packages:

```bash
cd frontend
npm install
```

### Step 3.2: Run the Frontend server
Start the Vite development server:

```bash
npm run dev
```
The frontend should now be running on `http://localhost:5173` (or whatever port Vite specifies). Open this URL in your browser to interact with the UI.

---

## 4. Admin Analytics Dashboard

We have a dedicated Streamlit dashboard for viewing system analytics, conversion rates, AI health, and demand-supply gaps.

### Step 4.1: Aggregating Analytics
To ensure the dashboard displays the latest data, analytics need to be aggregated from the raw events.
In a new terminal (with your python `venv` activated), run:

```bash
cd backend
python tools/analytics_aggregation.py today
```
*Note: Run this whenever you generate new data and want to view it in the dashboard.*

### Step 4.2: Run the Dashboard
Start the Streamlit server:

```bash
streamlit run admin/dashboard.py
```
This will open the dashboard in your browser (typically `http://localhost:8501`). Wait for it to load and enter the password specified in your `.env` file (`DASHBOARD_PASSWORD`).

---

## Architecture Quick Overview
- **YOLOv8 Model**: Detects clothing items bounds.
- **AGMAN Model**: Extracts attributes and creates 512-d embeddings.
- **LLM Engine**: Uses Groq/Gemini to translate natural language into SQL filtering criteria.
- **Supabase PostgreSQL**: Hosts the vector embeddings (FAISS/pgvector) and analytics events.
