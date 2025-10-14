#!/usr/bin/env python3

import json
from textwrap import dedent
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import re
import sys

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
        
        print("✓ OpenAI API key configured successfully")
    
    def create_grading_prompt(self, flattened_questions, answers_by_question):
        """Create a comprehensive prompt for GPT-4 to grade the answer"""
        prompt = f"""You are an expert grader for computer science assignments. Your task is to objectively grade a student's answer based on the provided criteria.

            **Questions:**
            {flattened_questions}

            **Students' Answers:**
            {answers_by_question}

            **Instructions:**
            1. Carefully analyze the students' answers against the grading criteria
            2. Compare it with the sample answer to understand the expected level of detail and accuracy
            3. Award points based on how well each student's answer meets the grading criteria
            4. Be objective and fair - partial credit should be given for partially correct answers
            5. Focus on the content and understanding demonstrated, not just exact wording
            6. Consider technical accuracy, completeness, and clarity of explanation

            **Response Format:**
            Provide your response as a JSON object with the following structure:
            {{
                "question_id": {{
                    "student": {{
                        "score": [numerical score out of points],
                        "breakdown": "Detailed explanation of how points were awarded for the student's answer",
                        "strengths": "What the student did well",
                        "areas_for_improvement": "What could be improved"
                    }},
                    ...
                }},
                ...
            }}

            Grade the answer now:"""
        return prompt

    def grade_assignment(self, assignment_json, submissions, max_retries=3):
        def flatten_questions(assignment_json):
            flattened_questions = []
            for question in assignment_json:
                if question['type'] == 'multi-part':
                    for subquestion in question['subquestions']:
                        if subquestion['type'] == 'multi-part':
                            for subsubquestion in subquestion['subquestions']:
                                flattened_questions.append(subsubquestion)
                        else:
                            flattened_questions.append(subquestion)
                else:
                    flattened_questions.append(question)
            return flattened_questions
        
        flattened_questions = flatten_questions(assignment_json)
        # print("flattened_questions", flattened_questions)
        # print("submissions", submissions)

        answers_by_question = {}
        for question in flattened_questions:
            for student, submission in submissions.items():
                for answer in submission['answer_sheet']:
                    if answer['question_number'] == str(question['id']):
                        if str(question['id']) not in answers_by_question:
                            answers_by_question[str(question['id'])] = []
                        answers_by_question[str(question['id'])].append({"student": student, **answer})
        
        # print("answers_by_question", answers_by_question)
        prompt = self.create_grading_prompt(flattened_questions, answers_by_question)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert academic grader. Provide objective, fair, and detailed grading based on the given criteria. Always respond with valid JSON."},
                        {"role": "user", "content": dedent(prompt)}
                    ],
                    temperature=0.1,  # Low temperature for consistent grading
                    # max_tokens=100000
                )

                result_text = response.choices[0].message.content.strip()
                print("Result text:", result_text)
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON pattern in the response
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        results_json = json.loads(json_match.group())
                        return results_json
                        # Validate the response has required fields
                        # if any('question_id' in key for key in results_json.keys()):
                        #     return results_json
                        # else:
                        #     raise ValueError("Response missing required 'question_id' field")
                    else:
                        raise ValueError("No JSON found in response")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    # If JSON parsing fails
                    if attempt == max_retries - 1:
                        raise ValueError(f"Could not extract score from response: {e}")
                    print(f"Warning: Failed to parse response (attempt {attempt + 1}): {e}")
                    continue
                    
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Return a default response if all attempts fail
                    return {
                        "score": 0,
                        "breakdown": f"Error in grading: {str(e)}",
                        "strengths": "Unable to evaluate due to API error",
                        "areas_for_improvement": "Unable to evaluate due to API error"
                    }
                time.sleep(2 ** attempt)  # Exponential backoff


        return results_json

def main():
    llm_grader = LLMGrader()
    assignment_json = json.load(open('Assignments/assignment.json'))
    submissions = {
        'student_1': json.load(open('Assignments/assignment_submissions/assignment_submission_1.json')),
        'student_2': json.load(open('Assignments/assignment_submissions/assignment_submission_2.json')),
        'student_3': json.load(open('Assignments/assignment_submissions/assignment_submission_3.json')),
    }
    results =   llm_grader.grade_assignment(assignment_json, submissions)
    print("Results", results)
    with open('assignment_wise_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()