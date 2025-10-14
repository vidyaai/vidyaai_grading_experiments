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
            Provide your response as a JSON object with the following structure:
            {{
                "score": [numerical score out of {full_points}],
                "breakdown": "Detailed explanation of how points were awarded for each criterion",
                "strengths": "What the student did well",
                "areas_for_improvement": "What could be improved"
            }}

            Grade the answer now:"""
        return prompt
    
    def grade_answer(self, question, student_answer, sample_answer, criteria, full_points, max_retries=3):
        # return {
        #     "score": full_points,
        #     "question": question,
        #     "student_answer": student_answer,
        #     "sample_answer": sample_answer
        # }

        """Grade a single answer using GPT-4"""
        prompt = self.create_grading_prompt(question, student_answer, sample_answer, criteria, full_points)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are an expert academic grader. Provide objective, fair, and detailed grading based on the given criteria. Always respond with valid JSON."},
                        {"role": "user", "content": dedent(prompt)}
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
                        # Validate the response has required fields
                        if 'score' in result_json:
                            return result_json
                        else:
                            raise ValueError("Response missing required 'score' field")
                    else:
                        raise ValueError("No JSON found in response")
                        
                except (json.JSONDecodeError, ValueError) as e:
                    # If JSON parsing fails, try to extract score manually
                    score_match = re.search(r'(?:score|points?)[\s:]*(\d+(?:\.\d+)?)', result_text, re.IGNORECASE)
                    if score_match:
                        score = float(score_match.group(1))
                        return {
                            "score": min(score, full_points),  # Cap at full_points
                            "breakdown": result_text,
                            "strengths": "See breakdown for details",
                            "areas_for_improvement": "See breakdown for details"
                        }
                    else:
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
        
        return None
    
    def grade_question(self, question_json, answers):
        results = {}
        for answer in answers:
            print("Grading answer:", answer['answer'])
            if not results.get(answer['student']):
                results[answer['student']] = {}
            results[answer['student']] = (
                self.grade_answer(question_json['question'], answer['answer'], question_json['correctAnswer'], question_json['rubric'], question_json['points'])
            )
        # print("Results for question:", question_json['question'], results)
        return results

    def grade_assignment(self, assignment_json, submissions):
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

        results = {}
        for question in flattened_questions:
            print("Grading question:", question)
            # print("Answers for question:", answers_by_question[str(question['id'])])
            if not str(question['id']) in answers_by_question:
                results[question['id']] = []
            results[question['id']] = self.grade_question(question, answers_by_question[str(question['id'])])
        # print("Results", results)
        return results

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
    with open('results.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()