import json
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import re

# Load environment variables
load_dotenv()

class LLMGrader:
    def __init__(self, api_key=None):
        """Initialize the LLM Grader with OpenAI API key"""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass it directly.")
            self.client = OpenAI(api_key=api_key)
    
    def create_grading_prompt(self, question, student_answer, sample_answer, criteria, full_points):
        """Create a comprehensive prompt for GPT-4 to grade the answer"""
        prompt = f"""You are an expert grader for computer science assignments. Your task is to objectively grade a student's answer based on the provided criteria.

**Question:**
{question}

**Student's Answer:**
{student_answer}

**Sample/Reference Answer:**
{sample_answer}

**Grading Criteria:**
{criteria}

**Total Points Available:** {full_points}

**Instructions:**
1. Carefully analyze the student's answer against the grading criteria
2. Compare it with the sample answer to understand the expected level of detail and accuracy
3. Award points based on how well the student's answer meets each criterion
4. Be objective and fair - partial credit should be given for partially correct answers
5. Focus on the content and understanding demonstrated, not just exact wording
6. Consider technical accuracy, completeness, and clarity of explanation

**Response Format:**
Provide your response in the following JSON format:
{{
    "score": [numerical score out of {full_points}],
    "breakdown": "Detailed explanation of how points were awarded for each criterion",
    "strengths": "What the student did well",
    "areas_for_improvement": "What could be improved"
}}

Grade the answer now:"""
        return prompt
    
    def grade_answer(self, question, student_answer, sample_answer, criteria, full_points, max_retries=3):
        """Grade a single answer using GPT-4"""
        prompt = self.create_grading_prompt(question, student_answer, sample_answer, criteria, full_points)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert academic grader. Provide objective, fair, and detailed grading based on the given criteria."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent grading
                    max_tokens=1000
                )
                
                result_text = response.choices[0].message.content.strip()
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON pattern in the response
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        result_json = json.loads(json_match.group())
                        return result_json
                    else:
                        # If no JSON found, try to parse the entire response
                        result_json = json.loads(result_text)
                        return result_json
                except json.JSONDecodeError:
                    # If JSON parsing fails, extract score manually and create a basic response
                    score_match = re.search(r'(?:score|points?)[\s:]*(\d+(?:\.\d+)?)', result_text, re.IGNORECASE)
                    if score_match:
                        score = float(score_match.group(1))
                        return {
                            "score": score,
                            "breakdown": result_text,
                            "strengths": "See breakdown",
                            "areas_for_improvement": "See breakdown"
                        }
                    else:
                        raise ValueError("Could not extract score from response")
                        
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Return a default response if all attempts fail
                    return {
                        "score": 0,
                        "breakdown": f"Error in grading: {str(e)}",
                        "strengths": "Unable to evaluate",
                        "areas_for_improvement": "Unable to evaluate"
                    }
                time.sleep(1)  # Wait before retry
    
    def load_question_data(self, question_folder):
        """Load grading data and criteria for a specific question"""
        grading_file = os.path.join(question_folder, 'grading.json')
        criteria_file = os.path.join('dataset_os', 'tutorialCriteria', f'{os.path.basename(question_folder)}.json')
        
        with open(grading_file, 'r', encoding='utf-8') as f:
            grading_data = json.load(f)
        
        with open(criteria_file, 'r', encoding='utf-8') as f:
            criteria_data = json.load(f)
        
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
                
                print(f"Grading Q{question_num} - Student {student_id}, Sub-question {sub_question_id}")
                
                # Grade using LLM
                llm_result = self.grade_answer(
                    question_text, 
                    student_answer, 
                    sample_answer, 
                    sample_criteria, 
                    full_points
                )
                
                results.append({
                    'Student_ID': student_id,
                    'Sub_Question_ID': sub_question_id,
                    'Question': question_text,
                    'Student_Answer': student_answer,
                    'Sample_Answer': sample_answer,
                    'Grading_Criteria': sample_criteria,
                    'Full_Points': full_points,
                    'Score_1': score_1,
                    'Score_2': score_2,
                    'Score_3': score_3,
                    'LLM_Score': llm_result.get('score', 0),
                    'LLM_Breakdown': llm_result.get('breakdown', ''),
                    'LLM_Strengths': llm_result.get('strengths', ''),
                    'LLM_Areas_for_Improvement': llm_result.get('areas_for_improvement', '')
                })
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
        
        return results
    
    def create_excel_report(self, results, question_num, output_dir='grading_results'):
        """Create an Excel report for a question"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Create Excel file with multiple sheets if needed
        output_file = os.path.join(output_dir, f'Q{question_num}_grading_results.xlsx')
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main results sheet
            df_main = df[['Student_ID', 'Sub_Question_ID', 'Full_Points', 'Score_1', 'Score_2', 'Score_3', 'LLM_Score']].copy()
            df_main.to_excel(writer, sheet_name='Scores', index=False)
            
            # Detailed analysis sheet
            df_detailed = df[['Student_ID', 'Sub_Question_ID', 'Student_Answer', 'LLM_Score', 'LLM_Breakdown', 'LLM_Strengths', 'LLM_Areas_for_Improvement']].copy()
            df_detailed.to_excel(writer, sheet_name='Detailed_Analysis', index=False)
            
            # Summary statistics sheet
            summary_stats = {
                'Metric': ['Average Score_1', 'Average Score_2', 'Average Score_3', 'Average LLM_Score', 
                          'Std Dev Score_1', 'Std Dev Score_2', 'Std Dev Score_3', 'Std Dev LLM_Score',
                          'Correlation Score_1 vs LLM', 'Correlation Score_2 vs LLM', 'Correlation Score_3 vs LLM'],
                'Value': [
                    df_main['Score_1'].mean(),
                    df_main['Score_2'].mean(),
                    df_main['Score_3'].mean(),
                    df_main['LLM_Score'].mean(),
                    df_main['Score_1'].std(),
                    df_main['Score_2'].std(),
                    df_main['Score_3'].std(),
                    df_main['LLM_Score'].std(),
                    df_main['Score_1'].corr(df_main['LLM_Score']),
                    df_main['Score_2'].corr(df_main['LLM_Score']),
                    df_main['Score_3'].corr(df_main['LLM_Score'])
                ]
            }
            pd.DataFrame(summary_stats).to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        print(f"Excel report saved: {output_file}")
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
                
                print(f"Completed Question {question_num}: {len(results)} answers graded")
                
            except Exception as e:
                print(f"Error processing Question {question_num}: {str(e)}")
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
                    'LLM_vs_Score1_Correlation': df['Score_1'].corr(df['LLM_Score']),
                    'LLM_vs_Score2_Correlation': df['Score_2'].corr(df['LLM_Score']),
                    'LLM_vs_Score3_Correlation': df['Score_3'].corr(df['LLM_Score'])
                })
        
        summary_df = pd.DataFrame(summary_data)
        output_file = os.path.join(output_dir, 'Combined_Summary.xlsx')
        summary_df.to_excel(output_file, index=False)
        print(f"Combined summary report saved: {output_file}")


def main():
    """Main function to run the grading system"""
    print("LLM-Based Assignment Grader")
    print("=" * 40)
    
    # Get OpenAI API key
    api_key = input("Enter your OpenAI API key (or press Enter if set in environment): ").strip()
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide it when prompted.")
            return
    
    try:
        # Initialize grader
        grader = LLMGrader(api_key)
        
        # Change to the correct directory
        os.chdir('/Users/pingakshyagoswami/hw_grading/llm_grader')
        
        # Grade all questions
        results = grader.grade_all_questions()
        
        print("\n" + "=" * 50)
        print("GRADING COMPLETED!")
        print("=" * 50)
        print("Check the 'grading_results' folder for Excel reports.")
        print("Each question has its own Excel file with multiple sheets:")
        print("- Scores: Main scoring comparison")
        print("- Detailed_Analysis: LLM feedback and analysis")
        print("- Summary_Statistics: Statistical analysis")
        print("- Combined_Summary.xlsx: Overall summary across all questions")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
