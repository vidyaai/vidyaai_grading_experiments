#!/usr/bin/env python3

import json
import pandas as pd
import os
import sys
import random

class DemoGrader:
    """Demo grader that simulates LLM grading without requiring OpenAI API"""
    
    def __init__(self):
        print("Initializing Demo Grader (no API required)")
    
    def simulate_grade_answer(self, question, student_answer, sample_answer, criteria, full_points):
        """Simulate grading with random scores for demo purposes"""
        # Simulate some basic scoring logic based on answer length and content
        answer_length = len(student_answer.split())
        sample_length = len(sample_answer.split())
        
        # Base score calculation (simplified)
        if answer_length < 5:
            base_score = full_points * 0.2
        elif answer_length < sample_length * 0.5:
            base_score = full_points * 0.4
        elif answer_length < sample_length * 0.8:
            base_score = full_points * 0.6
        else:
            base_score = full_points * 0.8
        
        # Add some randomness
        variation = random.uniform(-0.2, 0.2) * full_points
        final_score = max(0, min(full_points, base_score + variation))
        
        return {
            "score": round(final_score, 1),
            "breakdown": f"Simulated grading based on answer completeness and content quality. Answer length: {answer_length} words.",
            "strengths": "Demonstrates understanding of key concepts" if final_score > full_points * 0.5 else "Shows basic attempt at answering",
            "areas_for_improvement": "Could provide more detailed explanations" if final_score < full_points * 0.8 else "Good overall response"
        }
    
    def load_question_data(self, question_folder):
        """Load grading data and criteria for a specific question"""
        grading_file = os.path.join(question_folder, 'grading.json')
        criteria_file = os.path.join('dataset_os', 'tutorialCriteria', f'{os.path.basename(question_folder)}.json')
        
        with open(grading_file, 'r', encoding='utf-8') as f:
            grading_data = json.load(f)
        
        try:
            with open(criteria_file, 'r', encoding='utf-8') as f:
                criteria_data = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Criteria file not found for {question_folder}")
            criteria_data = {}
        
        return grading_data, criteria_data
    
    def process_question(self, question_num):
        """Process all answers for a specific question"""
        question_folder = f'dataset_os/q{question_num}'
        grading_data, criteria_data = self.load_question_data(question_folder)
        
        results = []
        
        # Process each student's answer
        for student_id, student_data in grading_data.items():
            for sub_question_id, answer_data in student_data.items():
                question_text = ' '.join(answer_data.get('question', []))
                student_answer = answer_data.get('answer', '')
                sample_answer = answer_data.get('sample_answer', '')
                sample_criteria = answer_data.get('sample_criteria', '')
                full_points = answer_data.get('full_points', 0)
                
                # Get existing scores for comparison
                score_1 = answer_data.get('score_1', 0)
                score_2 = answer_data.get('score_2', 0)
                score_3 = answer_data.get('score_3', 0)
                
                print(f"Processing Q{question_num} - Student {student_id}, Sub-question {sub_question_id}")
                
                # Simulate grading
                llm_result = self.simulate_grade_answer(
                    question_text, 
                    student_answer, 
                    sample_answer, 
                    sample_criteria, 
                    full_points
                )
                
                results.append({
                    'Student_ID': student_id,
                    'Sub_Question_ID': sub_question_id,
                    'Question': question_text[:100] + "..." if len(question_text) > 100 else question_text,
                    'Student_Answer': student_answer[:200] + "..." if len(student_answer) > 200 else student_answer,
                    'Sample_Answer': sample_answer[:200] + "..." if len(sample_answer) > 200 else sample_answer,
                    'Grading_Criteria': sample_criteria[:200] + "..." if len(sample_criteria) > 200 else sample_criteria,
                    'Full_Points': full_points,
                    'Score_1': score_1,
                    'Score_2': score_2,
                    'Score_3': score_3,
                    'LLM_Score': llm_result.get('score', 0),
                    'LLM_Breakdown': llm_result.get('breakdown', ''),
                    'LLM_Strengths': llm_result.get('strengths', ''),
                    'LLM_Areas_for_Improvement': llm_result.get('areas_for_improvement', '')
                })
        
        return results
    
    def create_excel_report(self, results, question_num, output_dir='grading_results'):
        """Create an Excel report for a question"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Create Excel file with multiple sheets
        output_file = os.path.join(output_dir, f'Q{question_num}_grading_results.xlsx')
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main results sheet - simplified for better readability
            df_main = df[['Student_ID', 'Sub_Question_ID', 'Full_Points', 'Score_1', 'Score_2', 'Score_3', 'LLM_Score']].copy()
            df_main.to_excel(writer, sheet_name='Scores', index=False)
            
            # Detailed analysis sheet
            df_detailed = df[['Student_ID', 'Sub_Question_ID', 'Student_Answer', 'LLM_Score', 'LLM_Breakdown', 'LLM_Strengths', 'LLM_Areas_for_Improvement']].copy()
            df_detailed.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
            
            # Summary statistics sheet
            summary_stats = {
                'Metric': ['Total Students', 'Average Score_1', 'Average Score_2', 'Average Score_3', 'Average LLM_Score', 
                          'Std Dev Score_1', 'Std Dev Score_2', 'Std Dev Score_3', 'Std Dev LLM_Score',
                          'Correlation Score_1 vs LLM', 'Correlation Score_2 vs LLM', 'Correlation Score_3 vs LLM'],
                'Value': [
                    len(df_main),
                    df_main['Score_1'].mean(),
                    df_main['Score_2'].mean(),
                    df_main['Score_3'].mean(),
                    df_main['LLM_Score'].mean(),
                    df_main['Score_1'].std(),
                    df_main['Score_2'].std(),
                    df_main['Score_3'].std(),
                    df_main['LLM_Score'].std(),
                    df_main['Score_1'].corr(df_main['LLM_Score']) if len(df_main) > 1 else 0,
                    df_main['Score_2'].corr(df_main['LLM_Score']) if len(df_main) > 1 else 0,
                    df_main['Score_3'].corr(df_main['LLM_Score']) if len(df_main) > 1 else 0
                ]
            }
            pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Questions and criteria sheet
            df_questions = df[['Student_ID', 'Question', 'Sample_Answer', 'Grading_Criteria']].drop_duplicates().copy()
            df_questions.to_excel(writer, sheet_name='Questions_Criteria', index=False)
        
        print(f"✓ Excel report saved: {output_file}")
        return output_file
    
    def grade_all_questions(self, output_dir='grading_results'):
        """Grade all questions (Q1-Q6) and create Excel reports"""
        all_results = {}
        
        for question_num in range(1, 7):
            print(f"\n{'='*50}")
            print(f"Processing Question {question_num}")
            print(f"{'='*50}")
            
            try:
                results = self.process_question(question_num)
                all_results[f'Q{question_num}'] = results
                
                # Create Excel report for this question
                self.create_excel_report(results, question_num, output_dir)
                
                print(f"✓ Completed Question {question_num}: {len(results)} answers processed")
                
            except Exception as e:
                print(f"✗ Error processing Question {question_num}: {str(e)}")
                continue
        
        # Create a combined summary report
        self.create_combined_summary(all_results, output_dir)
        
        return all_results
    
    def create_combined_summary(self, all_results, output_dir):
        """Create a combined summary report across all questions"""
        summary_data = []
        
        for question, results in all_results.items():
            if results:
                df = pd.DataFrame(results)
                summary_data.append({
                    'Question': question,
                    'Total_Answers': len(results),
                    'Avg_Full_Points': df['Full_Points'].mean(),
                    'Avg_Score_1': df['Score_1'].mean(),
                    'Avg_Score_2': df['Score_2'].mean(),
                    'Avg_Score_3': df['Score_3'].mean(),
                    'Avg_LLM_Score': df['LLM_Score'].mean(),
                    'LLM_vs_Score1_Correlation': df['Score_1'].corr(df['LLM_Score']) if len(df) > 1 else 0,
                    'LLM_vs_Score2_Correlation': df['Score_2'].corr(df['LLM_Score']) if len(df) > 1 else 0,
                    'LLM_vs_Score3_Correlation': df['Score_3'].corr(df['LLM_Score']) if len(df) > 1 else 0
                })
        
        summary_df = pd.DataFrame(summary_data)
        output_file = os.path.join(output_dir, 'Combined_Summary.xlsx')
        summary_df.to_excel(output_file, index=False)
        print(f"✓ Combined summary report saved: {output_file}")

def main():
    """Main function to run the demo grading system"""
    print("LLM-Based Assignment Grader (Demo Mode)")
    print("=" * 50)
    print("This demo version simulates LLM grading without requiring OpenAI API")
    print("For actual LLM grading, use llm_grader.py with your OpenAI API key")
    print("=" * 50)
    
    try:
        # Change to the correct directory
        os.chdir('/Users/pingakshyagoswami/hw_grading/llm_grader')
        
        # Initialize demo grader
        grader = DemoGrader()
        
        # Grade all questions
        results = grader.grade_all_questions()
        
        print("\n" + "=" * 50)
        print("DEMO GRADING COMPLETED!")
        print("=" * 50)
        print("Check the 'grading_results' folder for Excel reports.")
        print("Each question has its own Excel file with multiple sheets:")
        print("- Scores: Main scoring comparison")
        print("- Detailed_Analysis: Simulated LLM feedback and analysis") 
        print("- Summary_Statistics: Statistical analysis")
        print("- Questions_Criteria: Questions and grading criteria")
        print("- Combined_Summary.xlsx: Overall summary across all questions")
        print("\nNote: Scores in this demo are simulated. For real LLM grading,")
        print("use the full version with OpenAI API key.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
