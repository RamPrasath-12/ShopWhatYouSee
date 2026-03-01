import psycopg2

conn = psycopg2.connect(
    "postgresql://postgres:ShopWhatYouSee123@db.mxjpueufbooxgewqxccm.supabase.co:5432/postgres"
)

print("Connected successfully!")
conn.close()
