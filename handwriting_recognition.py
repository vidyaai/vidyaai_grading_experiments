import re
import dotenv
dotenv.load_dotenv()

import os
from datetime import datetime
import glob
import base64
from openai import OpenAI
import json
import time

class HandwritingAnalyzer:
    def __init__(self, data_directory, text_output_dir="text_comparisons"):
        self.data_dir = data_directory
        self.text_output_dir = text_output_dir
        self.results = []
        
        # Create text output directory if it doesn't exist
        os.makedirs(self.text_output_dir, exist_ok=True)
        
        # Initialize OpenAI client with timeout settings
        self.client = OpenAI(
            timeout=60.0,  # 60 second timeout
        )
        print("OpenAI GPT-4o client initialized successfully!")

    def encode_image_to_base64(self, image_path):
        """Encode image to base64 for OpenAI API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")

    def extract_handwritten_text_with_retry(self, image_path, max_retries=3):
        """Extract text with retry logic for better reliability"""
        
        for attempt in range(max_retries):
            try:
                print(f"    Attempt {attempt + 1}/{max_retries}...")
                
                # Add delay between retries
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    print(f"    Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                
                # Try the extraction
                result = self.extract_handwritten_text_detailed_gpt4o(image_path)
                print(f"    Success on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"    Attempt {attempt + 1} failed: {e}")
                
                # Check if it's a rate limit or connection error
                if "rate limit" in error_msg or "429" in error_msg:
                    print(f"    Rate limit detected, waiting longer...")
                    time.sleep(10)  # Wait 10 seconds for rate limit
                elif "connection" in error_msg or "timeout" in error_msg:
                    print(f"    Connection issue detected...")
                    time.sleep(5)  # Wait 5 seconds for connection issues
                
                # If this is the last attempt, raise the error
                if attempt == max_retries - 1:
                    raise e
        
        # This shouldn't be reached, but just in case
        raise Exception(f"Failed after {max_retries} attempts")

    def extract_handwritten_text_detailed_gpt4o(self, image_path):
        """Extract text with detailed analysis using GPT-4o Vision"""
        
        try:
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image_path)
            
            # Create a more detailed prompt
            prompt = """
            You are an expert handwriting analyst. Please analyze this handwritten text image and provide a detailed extraction.

            Please provide your response in the following JSON format:
            {
                "extracted_text": "The complete text you can read from the image",
                "confidence_level": "high/medium/low",
                "difficult_words": ["list", "of", "words", "you", "are", "unsure", "about"],
                "notes": "Any additional observations about the handwriting quality, style, or challenges"
            }

            Instructions:
            1. Extract ALL visible text as accurately as possible
            2. Maintain original punctuation and spacing
            3. If unsure about specific words, include them in difficult_words
            4. Assess your overall confidence in the extraction
            5. Note any challenges (poor lighting, unclear handwriting, etc.)

            Analyze this handwritten image:
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            # print("response_text: ", response_text)
            # Try to parse JSON response
            try:
                analysis = self.extract_json(response_text)
                return analysis
            except json.JSONDecodeError:
                # If JSON parsing fails, return simple format
                return {
                    "extracted_text": response_text,
                    "confidence_level": "unknown",
                    "difficult_words": [],
                    "notes": "JSON parsing failed, returned raw response"
                }
            
        except Exception as e:
            raise ValueError(f"GPT-4o detailed handwriting extraction failed: {e}")

    def extract_handwritten_text_simple_fallback(self, image_path):
        """Simple fallback extraction method"""
        
        try:
            base64_image = self.encode_image_to_base64(image_path)
            
            prompt = "Please extract all the handwritten text from this image. Return only the text, no explanations."
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"  # Use low detail for faster processing
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content.strip()
            
            return {
                "extracted_text": extracted_text,
                "confidence_level": "unknown",
                "difficult_words": [],
                "notes": "Simple fallback extraction used"
            }
            
        except Exception as e:
            raise ValueError(f"Simple fallback extraction failed: {e}")

    def find_image_files(self):
        """Find .jpg files"""
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Directory '{self.data_dir}' not found!")
            return []
        
        # Find all jpg files
        jpg_pattern = os.path.join(self.data_dir, "*.jpg")
        jpg_files = glob.glob(jpg_pattern)
        
        # Find matching image files
        matching_images = []
        
        for jpg_file in jpg_files:
            # Get base name without extension
            base_name = os.path.splitext(os.path.basename(jpg_file))[0]

            if os.path.exists(jpg_file):
                matching_images.append({
                    'jpg_file': jpg_file,
                    'base_name': base_name
                })
            else:
                print(f"Warning: No matching .jpg file found for {os.path.basename(jpg_file)}")
        
        return matching_images

    def extract_json(self, text: str):
        # Grab first fenced code block (```json ... ``` or ``` ... ```)
        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.I)
        candidates = blocks or [text]  # fallback: try whole text

        for c in candidates:
            c = c.strip()
            # Try direct parse
            try:
                return json.loads(c)
            except Exception:
                # Fallback: parse the largest {...} span inside
                start, end = c.find("{"), c.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(c[start:end+1])
                    except Exception:
                        pass
        raise ValueError("No valid JSON found in text")

    def analyze_file(self, jpg_file):
        """Analyze .jpg file with robust error handling"""
        
        print(f"Processing: {jpg_file}")
        
        try:
            # Extract handwritten text using GPT-4o with retry logic
            print(f"Analyzing handwriting with GPT-4o...")
            
            try:
                # Try detailed extraction first
                handwritten_analysis = self.extract_handwritten_text_with_retry(jpg_file, max_retries=3)
            except Exception as e:
                print(f"Detailed extraction failed, trying simple fallback: {e}")
                try:
                    # Try simple fallback
                    handwritten_analysis = self.extract_handwritten_text_simple_fallback(jpg_file)
                except Exception as e2:
                    print(f"All extraction methods failed: {e2}")
                    raise e2
            
            print("handwritten_analysis: ", handwritten_analysis)
            # Extract the text from analysis
            if isinstance(handwritten_analysis, dict):
                handwritten_text = handwritten_analysis.get('extracted_text', '')
                confidence = handwritten_analysis.get('confidence_level', 'unknown')
            else:
                handwritten_text = str(handwritten_analysis)
                confidence = 'unknown'
            
            # Prepare result dictionary
            result = {
                'jpg_file': os.path.basename(jpg_file),
                'gpt4o_confidence': confidence,  # New field for GPT-4o confidence
                'status': 'SUCCESS',
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'handwritten_text': handwritten_text
            }
            
            print(f"GPT-4o Confidence: {confidence}")
            
            return result
            
        except Exception as e:
            print(f"Error analyzing image file: {str(e)}")
            return {
                'jpg_file': os.path.basename(jpg_file) if 'jpg_file' in locals() else 'N/A',
                'gpt4o_confidence': 'error',
                'status': f'ERROR: {str(e)}',
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'handwritten_text': None
            }

    def process_all_image_files(self):
        """Process all matching .jpg file pairs"""
        
        # Find matching image files
        image_files = self.find_image_files()
        
        if not image_files:
            print(f"No .jpg files found in '{self.data_dir}' directory!")
            return False
        
        print(f"Found {len(image_files)} .jpg files to process")
        print(f"Using GPT-4o Vision API with retry logic for reliability")
        print("=" * 60)
        
        # Process each file pair
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing...")
            result = self.analyze_file(image_file['jpg_file'])
            self.results.append(result)
            
            # Add a small delay between requests to avoid rate limiting
            if i < len(image_files):  # Don't wait after the last file
                time.sleep(1)  # 1 second delay between requests
        
        print("\n" + "=" * 60)
        print(f"Processing complete! Processed {len(self.results)} image files")
        
        return True

    def save_to_text_file(self):
        """Save results to text file"""
        
        if not self.results:
            print("No results to save!")
            return False
        
        print(f"Saving results to text files: {self.text_output_dir}")

        for result in self.results:
            try:
                with open(self.text_output_dir + "/" + result['jpg_file'] + ".txt", "w", encoding='utf-8') as f:
                    print("handwritten text: ", result['handwritten_text'])
                    f.write(result['handwritten_text'])
            except Exception as e:
                print(f"Error saving to text file: {e}")
                return False

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        
        print("Handwriting Recognition Pipeline (GPT-4o Robust)")
        print("=" * 60)
        print(f"Data Directory: {self.data_dir}")
        print(f"Text Comparisons Directory: {self.text_output_dir}")
        print("OCR Method: GPT-4o Vision API with retry logic and fallbacks")
        print("Filetype: .jpg (handwritten)")
        print()
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("Error: OPENAI_API_KEY environment variable not set!")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            return
        
        # Process all image files
        if self.process_all_image_files():
            # Print summary
            successful = len([r for r in self.results if r['status'] == 'SUCCESS'])
            failed = len(self.results) - successful
            
            print(f"\nANALYSIS SUMMARY:")
            print(f"Total file pairs processed: {len(self.results)}")
            print(f"Successful analyses: {successful}")
            print(f"Failed analyses: {failed}")
            print(f"Success rate: {(successful/len(self.results)*100):.1f}%")

            # Save to text file
            self.save_to_text_file()
            
        else:
            print("Analysis failed!")

def main():
    """Main function"""
    
    # Configuration
    data_directory = "Biology-0089-0159"  # Biology directory
    text_output_directory = "biology_text_comparisons"  # Directory for individual text files

    # Create analyzer and run
    analyzer = HandwritingAnalyzer(data_directory, text_output_directory)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()