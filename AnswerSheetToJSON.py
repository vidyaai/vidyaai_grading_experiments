import dotenv
dotenv.load_dotenv()

import os
import glob
import base64
from openai import OpenAI
import json
from textwrap import dedent
from io import BytesIO
from typing import List

from pdf2image import convert_from_path

class AnswerSheetToJSON:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
        self.client = OpenAI()
        print("OpenAI GPT-4o client initialized successfully!")
        self.results = []

    def process_all_pdfs(self):
        pdf_files = glob.glob(os.path.join(self.input_dir, "*.pdf"))
        for i, pdf_file in enumerate(pdf_files):
            print(f"Processing {i+1} of {len(pdf_files)}: {pdf_file}")
            try:
                self.process_pdf_with_diagram(pdf_file)
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
                continue

    def process_pdf_with_diagram(self, pdf_file):
        print(f"Processing: {pdf_file}")

        # 1) Convert each PDF page to PNG (requires Poppler on Windows)
        poppler_path = os.getenv("POPPLER_PATH", None)
        print("Converting PDF to images...")
        try:
            pages = convert_from_path(pdf_file, dpi=200, poppler_path=poppler_path)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images. Ensure Poppler is installed and POPPLER_PATH is set. Original error: {e}")

        def pil_image_to_data_url(img) -> str:
            buffered = BytesIO()
            img.save(buffered, format="jpeg")
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        print("Converted PDF to images successfully!")

        print("Building multimodal prompt...")
        image_contents: List[dict] = []
        for page_num, page_img in enumerate(pages, 1):
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": pil_image_to_data_url(page_img)}
            })
        print("Built multimodal prompt successfully!")

        print("Building system prompt...")
        # 2) Build multimodal prompt with bounding box extraction
        system_prompt = """
            You are a precise assistant that converts answer sheets into structured JSON.
            Return only JSON, adhering strictly to the provided schema.
        """
        print("Built system prompt successfully!")

        print("Building prompt...")
        prompt = f"""
            Convert the following answer sheet images into a JSON object. The images are provided in order: Page 1, Page 2, Page 3, etc.

            Example JSON structure:
            {{
            "answer_sheet": [
                {{
                "question_number": "1.1",
                "answer": "Applying coating of zinc",
                "diagram": {{
                    "label": "Circuit diagram for Q2",
                    "bounding_box": {{"x": 120, "y": 240, "width": 460, "height": 320, "page_number": 1}}
                }}
                }},
                {{
                "question_number": "1.2",
                "answer": "increases",
                "diagram": null
                }}
            ]
            }}

            Rules:
            - DO NOT MAKE ANYTHING UP. DO NOT INCLUDE ANYTHING OTHER THAN THE ANSWER SHEET.
            - TRY TO INCLUDE THE FULL ANSWER FROM THE ANSWER SHEET.
            - Flatten multipart questions completely (e.g., 1.i → 1.1, 2.a → 2.1).
            - Each item must include:
            - question_number: string
            - answer: extracted answer from the answer sheet
            - diagram: null if no diagram; otherwise include label and a tight bounding_box around the drawn diagram for that question only.
            - Bounding boxes must be integers in image pixel coordinates of the provided JPEGs (top-left origin).
            - CRITICAL: Each bounding_box must include a "page_number" field indicating which page (1, 2, 3, etc.) the diagram appears on.
            - If a diagram is present but unlabeled, just put "unlabelled" in the label field.
            - Preserve question order and ignore non-answer metadata.
            - Return only valid JSON that conforms to the schema.
        """
        print("Built prompt successfully!")

        print("Building JSON schema...")
        # 3) JSON schema including optional diagram with bounding box
        answer_sheet_json = {
            "name": "answer_sheet_json",
            "type": "object",
            "properties": {
                "answer_sheet": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_number": {"type": "string"},
                            "answer": {"type": "string"},
                            "diagram": {
                                "oneOf": [
                                    {"type": "null"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string"},
                                            "bounding_box": {
                                                "type": "object",
                                                "properties": {
                                                    "x": {"type": "integer"},
                                                    "y": {"type": "integer"},
                                                    "width": {"type": "integer"},
                                                    "height": {"type": "integer"},
                                                    "page_number": {"type": "integer"}
                                                },
                                                "required": ["x", "y", "width", "height", "page_number"]
                                            }
                                        },
                                        "required": ["label", "bounding_box"]
                                    }
                                ]
                            }
                        },
                        "required": ["question_number", "answer", "diagram"]
                    }
                }
            },
            "required": ["answer_sheet"]
        }
        print("Built JSON schema successfully!")

        print("Calling GPT-5 with multimodal content...")
        # 4) Call GPT-5 with multimodal content
        try:
            completion = self.client.chat.completions.parse(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": dedent(system_prompt)},
                    {"role": "user", "content": [
                        {"type": "text", "text": dedent(prompt)},
                        *image_contents
                    ]},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": answer_sheet_json["name"],
                        "schema": answer_sheet_json,
                    },
                }
            )
            print("Called GPT-5 with multimodal content successfully!")
        except Exception as e:
            print(f"Error calling GPT-5 with multimodal content: {e}")
            raise e

        print("Writing output to file...")
        output_path = os.path.join(self.output_dir, os.path.basename(pdf_file).replace(".pdf", ".json"))
        try:
            print(completion.choices[0].message.content)
        except UnicodeEncodeError:
            print("Output contains Unicode characters that cannot be displayed in console")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(completion.choices[0].message.content)
            print(f"Written output to file {output_path} successfully!")
        except Exception as e:
            print(f"Error writing output to file: {e}")
            raise e
        try:
            result = {
                "answer_sheet": completion.choices[0].message.content,
                "pdf_file": pdf_file
            }
            self.results.append(result)
        except Exception as e:
            print(f"Error writing output to file: {e}")
            raise e

    def process_pdf_wo_diagram(self, pdf_file):
        print(f"Processing: {pdf_file}")
        file = self.client.files.create(
            file=open(pdf_file, "rb"),
            purpose="user_data"
        )
        system_prompt = """
        You are a precise and detail-oriented assistant that converts answer sheets from PDFs into structured JSON.
        All answers must be returned in a *flat JSON array* — no nesting allowed.
        """

        prompt = """
        Convert the following PDF answer sheet into a JSON object with this structure:
        {
        "answer_sheet": [
            {
            "question_number": "1.1",
            "answer": "Answer text here"
            },
            ...
        ]
        }

        ### Rules:
        1. **Flatten multipart questions** completely.
        - For example:
            - 1.i, 1.ii, 1.iii → 1.1, 1.2, 1.3
            - 2.a, 2.b, 2.c → 2.1, 2.2, 2.3
            - 1.1.1 → keep as "1.1.1" (do not nest).
        2. **Do not use nested structures or arrays inside answers** — every answer should be at the top level.
        3. Each item must have:
        - `question_number`: string (like "1.1" or "2.3.1")
        - `answer`: string (clean and concise answer text)
        4. Preserve numbering order as it appears in the sheet.
        5. Ignore any non-answer text, metadata, or page numbers.
        6. Ensure consistent use of dots for hierarchy — replace any letters or roman numerals with numeric equivalents.

        Return only valid JSON conforming to the schema.
        """
        answer_sheet_json = {
            "name": "answer_sheet_json",
            "type": "object",
            "properties": {
                "answer_sheet": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_number": {
                                "type": "string"
                            },
                            "answer": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["question_number", "answer"]
                }
            },
            "required": ["answer_sheet"]
        }
        completion = self.client.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": dedent(system_prompt)},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": dedent(prompt)
                    },
                    {
                        "type": "file",
                        "file": {
                            "file_id": file.id
                        }
                    }
                ]}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": answer_sheet_json["name"],
                    "schema": answer_sheet_json,
                },
            }
        )
        with open(os.path.join(self.output_dir, os.path.basename(pdf_file).replace(".pdf", ".json")), "w", encoding="utf-8") as f:
            f.write(completion.choices[0].message.content)

    def run_analysis(self):
        self.process_all_pdfs()
        with open(os.path.join(self.output_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)
        print("Written results to file successfully!")
        return self.results

def main():
    """Main function"""
    
    # Configuration
    input_dir = "pdf_input"
    output_dir = "json_output_of_pdf_answer_sheet"
    
    answer_sheet_to_json = AnswerSheetToJSON(input_dir=input_dir, output_dir=output_dir)
    answer_sheet_to_json.run_analysis()

if __name__ == "__main__":
    main()
