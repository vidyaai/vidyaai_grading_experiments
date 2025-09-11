#!/bin/bash

# LLM Grader Launcher Script

echo "LLM-Based Assignment Grader"
echo "=========================="

# Check if we're in the right directory
if [ ! -d "llm_grader" ]; then
    echo "Error: Please run this script from the hw_grading directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "grading_env" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv grading_env
    source grading_env/bin/activate
    pip install -r requirements.txt
else
    echo "Activating virtual environment..."
    source grading_env/bin/activate
fi

# Change to the data directory
cd llm_grader

echo ""
echo "Choose an option:"
echo "1. Run demo grader (no API key required)"
echo "2. Run real LLM grader (requires OpenAI API key)"
echo "3. Test setup"
echo "4. View results folder"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Running demo grader..."
        python ../demo_grader.py
        ;;
    2)
        echo "Running real LLM grader..."
        python ../real_llm_grader.py
        ;;
    3)
        echo "Testing setup..."
        python ../test_setup.py
        ;;
    4)
        echo "Opening results folder..."
        if [ -d "grading_results" ]; then
            open grading_results
        else
            echo "No results folder found. Run the grader first."
        fi
        ;;
    *)
        echo "Invalid choice."
        ;;
esac

echo "Done!"
