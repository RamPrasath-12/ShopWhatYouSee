# Quick script to check database build progress

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "PRODUCT DATABASE BUILD - PROGRESS CHECK" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Activate venv and check progress
.\venv\Scripts\Activate.ps1

# Count completed products
$result = python -c "import sqlite3; conn = sqlite3.connect('../data/products.db'); c = conn.cursor(); c.execute('SELECT COUNT(*) FROM products WHERE processing_status=''completed'''); print(c.fetchone()[0])"

Write-Host "Completed Products: $result / 34,215" -ForegroundColor Green

# Calculate percentage
$percent = [math]::Round(($result / 34215) * 100, 2)
Write-Host "Progress: $percent%" -ForegroundColor Yellow

# Check last 10 log entries
Write-Host ""
Write-Host "Last 10 Log Entries:" -ForegroundColor Cyan
Get-Content "build_database.log" -Tail 10

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
