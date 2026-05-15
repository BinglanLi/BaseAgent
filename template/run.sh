#!/bin/bash
# Run the KG pipeline

echo "================================"
echo "KG Pipeline"
echo "================================"
echo ""

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "No virtual environment found. Run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# Run the pipeline
python src/main.py "$@"

echo ""
echo "================================"
echo "Done."
echo "================================"
