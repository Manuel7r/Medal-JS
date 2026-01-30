#!/bin/bash
# Start Medal Trading System (backend + frontend)
cd "$(dirname "$0")"

echo "=== Starting Medal Trading System ==="

# Kill any existing processes on our ports
lsof -ti :8000 | xargs kill -9 2>/dev/null
lsof -ti :5180 | xargs kill -9 2>/dev/null
sleep 1

# Start backend
echo "Starting backend on :8000..."
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on :5180..."
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "=== Medal Trading System Running ==="
echo "  Frontend: http://localhost:5180"
echo "  Backend:  http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers"

# Trap Ctrl+C to kill both
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
