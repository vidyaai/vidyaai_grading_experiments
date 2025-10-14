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
    
    def _flatten_questions(self, assignment_json):
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

    def create_grading_prompt(self, flattened_questions, student_id, student_answersheet):
        """Create a comprehensive prompt for GPT-4 to grade one student's entire assignment"""
        prompt = f"""You are an expert grader for computer science assignments. Your task is to objectively grade one student's entire assignment based on the provided questions, sample answers, and rubrics.

            **Questions (flattened):**
            {flattened_questions}

            **Student ID:** {student_id}

            **Student's Answer Sheet:**
            {student_answersheet}

            Each question object includes: id, question, correctAnswer, rubric, points.
            Each answer entry includes: question_number (string), answer (string).

            **Instructions:**
            1. For each question id in Questions, find the matching answer by `question_number` in the student's answer sheet.
            2. Carefully analyze the student's answer against the question's rubric and compare with the sample (correctAnswer).
            3. Award points based on how well the student's answer meets each criterion. Give partial credit where reasonable.
            4. Focus on correctness, completeness, clarity, and depth.
            5. If the student did not answer a question, assign 0 and say so in the breakdown.

            **Response Format:**
            Provide your response as a JSON object where each key is the question id (as a string), and the value is an object with:
            {{
                "score": [numerical score out of that question's `points`],
                "breakdown": "Explanation of how points were assigned per rubric",
                "strengths": "What the student did well",
                "areas_for_improvement": "What can be improved"
            }}

            Only output JSON with no extra commentary.

            Grade now:"""
        return prompt
    
    def grade_student(self, assignment_json, student_id, student_submission, max_retries=3):
        flattened_questions = self._flatten_questions(assignment_json)
        prompt = self.create_grading_prompt(flattened_questions, student_id, student_submission.get('answer_sheet', []))

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert academic grader. Provide objective, fair, and detailed grading based on the given criteria. Always respond with valid JSON."},
                        {"role": "user", "content": dedent(prompt)}
                    ],
                    temperature=0.1,
                    max_tokens=10000
                )

                result_text = response.choices[0].message.content.strip()

                # Try to extract JSON from the response
                try:
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        results_json = json.loads(json_match.group())
                        # Basic validation: keys should look like question ids
                        if isinstance(results_json, dict) and len(results_json.keys()) > 0:
                            return results_json
                        else:
                            raise ValueError("Parsed JSON does not look like a mapping of question results")
                    else:
                        raise ValueError("No JSON found in response")
                except (json.JSONDecodeError, ValueError) as e:
                    if attempt == max_retries - 1:
                        raise ValueError(f"Could not parse grading response: {e}")
                    print(f"Warning: Failed to parse response (attempt {attempt + 1}): {e}")
                    continue

            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    # Return a default response if all attempts fail
                    return {
                        "error": f"Error in grading: {str(e)}"
                    }
                time.sleep(2 ** attempt)  # Exponential backoff

    def grade_assignment(self, assignment_json, submissions):
        """Grade the assignment student-wise: one LLM call per student"""
        results_by_student = {}
        for student_id, student_submission in submissions.items():
            print("Grading student:", student_id)
            results_by_student[student_id] = self.grade_student(assignment_json, student_id, student_submission)
        return results_by_student

def main():
    llm_grader = LLMGrader()
    assignment_json = json.load(open('Assignments/assignment.json'))
    submissions = {
        'student_1': json.load(open('Assignments/assignment_submissions/assignment_submission_1.json')),
        'student_2': json.load(open('Assignments/assignment_submissions/assignment_submission_2.json')),
        'student_3': json.load(open('Assignments/assignment_submissions/assignment_submission_3.json')),
    }
    results = llm_grader.grade_assignment(assignment_json, submissions)
    print("Results", results)
    with open('student_wise_results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()

