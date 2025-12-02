#!/bin/bash
echo "Starting PCB Defect Detection App..."

# Check if best.pt exists
if [ ! -f "backend/best.pt" ]; then
    echo "⚠️  WARNING: 'backend/best.pt' not found!"
    echo "Please place your trained YOLO model file named 'best.pt' in the 'backend' directory."
    echo "The app will start, but prediction will fail without the model."
    echo "---------------------------------------------------"
fi

# Install dependencies
echo "Installing dependencies..."
python3 -m pip install -r backend/requirements.txt

# Run the server
echo "Starting FastAPI server..."
echo "Open http://localhost:8000 in your browser."
cd backend
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
