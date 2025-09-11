#!/usr/bin/env python3

import json
import pandas as pd
import os
import sys

def test_data_loading():
    """Test loading the JSON data files"""
    print("Testing data loading...")
    
    # Change to the correct directory
    os.chdir('/Users/pingakshyagoswami/hw_grading/llm_grader')
    
    # Test loading Q1 data
    try:
        with open('dataset_os/q1/grading.json', 'r', encoding='utf-8') as f:
            grading_data = json.load(f)
        
        with open('dataset_os/tutorialCriteria/q1.json', 'r', encoding='utf-8') as f:
            criteria_data = json.load(f)
        
        print(f"✓ Successfully loaded Q1 data")
        print(f"  - Number of students: {len(grading_data)}")
        print(f"  - First student data keys: {list(grading_data['1'].keys())}")
        
        # Show sample data structure
        first_student = list(grading_data.keys())[0]
        first_question = list(grading_data[first_student].keys())[0]
        sample_answer = grading_data[first_student][first_question]
        
        print(f"  - Sample answer structure: {list(sample_answer.keys())}")
        print(f"  - Full points for Q1: {sample_answer.get('full_points', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        return False

def create_sample_excel():
    """Create a sample Excel file with the expected structure"""
    print("\nCreating sample Excel structure...")
    
    try:
        # Sample data
        sample_data = {
            'Student_ID': ['1', '2', '3', '4'],
            'Sub_Question_ID': ['1', '1', '1', '1'],
            'Full_Points': [19, 19, 19, 19],
            'Score_1': [7.0, 19.0, 13.0, 16.0],
            'Score_2': [7.0, 19.0, 13.0, 16.0],
            'Score_3': [7.0, 19.0, 19.0, 16.0],
            'LLM_Score': [0, 0, 0, 0]  # Placeholder for LLM scores
        }
        
        df = pd.DataFrame(sample_data)
        
        # Create output directory
        output_dir = 'sample_output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save to Excel
        output_file = os.path.join(output_dir, 'sample_grading_structure.xlsx')
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Scores', index=False)
        
        print(f"✓ Sample Excel file created: {output_file}")
        return True
        
    except Exception as e:
        print(f"✗ Error creating Excel: {str(e)}")
        return False

def main():
    print("LLM Grader Setup Test")
    print("=" * 30)
    
    # Test data loading
    if not test_data_loading():
        print("Data loading failed. Please check file paths.")
        return False
    
    # Test Excel creation
    if not create_sample_excel():
        print("Excel creation failed. Please check pandas/openpyxl installation.")
        return False
    
    print("\n" + "=" * 30)
    print("✓ All tests passed!")
    print("You can now run the full grading script.")
    print("Make sure to set your OpenAI API key before running:")
    print("export OPENAI_API_KEY='your-api-key-here'")
    
    return True

if __name__ == "__main__":
    main()
