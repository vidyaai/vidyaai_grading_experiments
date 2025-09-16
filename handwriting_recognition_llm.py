import cv2
import re
import difflib
from collections import Counter
import os
import pandas as pd
from datetime import datetime
import glob
from pathlib import Path
import base64
from openai import OpenAI
import json
import time
import requests

class BiologyHandwritingAnalyzerGPT4O:
    def __init__(self, data_directory="Biology-0089-059", output_file="biology_handwriting_analysis_results_gpt4o.xlsx", text_output_dir="biology_text_comparisons_gpt4o"):
        self.data_dir = data_directory
        self.output_file = output_file
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
                print(f"    ✅ Success on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"    ❌ Attempt {attempt + 1} failed: {e}")
                
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
            
            # Try to parse JSON response
            try:
                analysis = json.loads(response_text)
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

    def read_printed_text(self, text_path):
        """Read printed text from .txt file"""
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                printed_text = f.read().strip()
            return printed_text
        except Exception as e:
            raise ValueError(f"Could not read text file {text_path}: {e}")

    def clean_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase and remove extra spaces
        text = re.sub(r'\s+', ' ', text.lower().strip())
        return text

    def find_matching_files(self):
        """Find matching .jpg and .txt file pairs"""
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Directory '{self.data_dir}' not found!")
            return []
        
        # Find all jpg files
        jpg_pattern = os.path.join(self.data_dir, "*.jpg")
        jpg_files = glob.glob(jpg_pattern)
        
        # Find matching txt files
        matching_pairs = []
        
        for jpg_file in jpg_files:
            # Get base name without extension
            base_name = os.path.splitext(os.path.basename(jpg_file))[0]
            
            # Look for corresponding txt file
            txt_file = os.path.join(self.data_dir, f"{base_name}.txt")
            
            if os.path.exists(txt_file):
                matching_pairs.append({
                    'jpg_file': jpg_file,
                    'txt_file': txt_file,
                    'base_name': base_name
                })
            else:
                print(f"Warning: No matching .txt file found for {os.path.basename(jpg_file)}")
        
        return matching_pairs

    def save_text_comparison(self, base_name, printed_text, handwritten_analysis, comparison_results):
        """Save text comparison to individual text file"""
        
        text_filename = f"{base_name}_comparison.txt"
        text_filepath = os.path.join(self.text_output_dir, text_filename)
        
        # Extract text and analysis details
        if isinstance(handwritten_analysis, dict):
            handwritten_text = handwritten_analysis.get('extracted_text', '')
            confidence = handwritten_analysis.get('confidence_level', 'unknown')
            difficult_words = handwritten_analysis.get('difficult_words', [])
            notes = handwritten_analysis.get('notes', '')
        else:
            handwritten_text = str(handwritten_analysis)
            confidence = 'unknown'
            difficult_words = []
            notes = 'Simple extraction mode'
        
        try:
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"TEXT COMPARISON FOR: {base_name}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"OCR Method: GPT-4o Vision API\n")
                f.write(f"Confidence Level: {confidence}\n\n")
                
                # Raw texts
                f.write("📄 ORIGINAL TEXTS:\n")
                f.write("-" * 50 + "\n")
                f.write("PRINTED TEXT (from .txt file):\n")
                f.write(f'"""{printed_text}"""\n\n')
                
                f.write("HANDWRITTEN TEXT (GPT-4o OCR from .jpg image):\n")
                f.write(f'"""{handwritten_text}"""\n\n')
                
                # GPT-4o Analysis Details
                if difficult_words:
                    f.write("🤔 GPT-4o ANALYSIS DETAILS:\n")
                    f.write("-" * 50 + "\n")
                    f.write(f"Difficult/Uncertain Words: {difficult_words}\n")
                    f.write(f"Analysis Notes: {notes}\n\n")
                
                # Analysis results
                f.write("📊 ANALYSIS RESULTS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Word Score: {comparison_results['word_score']} ({comparison_results['word_accuracy']}% accuracy)\n")
                f.write(f"Punctuation Score: {comparison_results['punct_score']} ({comparison_results['punct_accuracy']}% accuracy)\n")
                f.write(f"Overall Similarity: {comparison_results['overall_similarity']}%\n\n")
                
                # Footer
                f.write("=" * 80 + "\n")
                f.write("END OF COMPARISON\n")
                f.write("=" * 80 + "\n")
                
            print(f"  📝 Text comparison saved: {text_filepath}")
            return True
            
        except Exception as e:
            print(f"  ❌ Error saving text comparison: {e}")
            return False

    def compare_texts(self, printed_text, handwritten_text):
        """Compare printed text with handwritten text and return similarity scores"""
        
        # Clean texts
        clean_printed = self.clean_text(printed_text)
        clean_handwritten = self.clean_text(handwritten_text)
        
        # Extract words
        printed_words = re.findall(r'\w+', clean_printed)
        handwritten_words = re.findall(r'\w+', clean_handwritten)
        
        # Extract punctuation
        printed_punct = re.findall(r'[^\w\s]', clean_printed)
        handwritten_punct = re.findall(r'[^\w\s]', clean_handwritten)
        
        # Calculate word accuracy
        if printed_words:
            # Count exact word matches
            printed_counter = Counter(printed_words)
            handwritten_counter = Counter(handwritten_words)
            
            correct_words = 0
            for word, count in printed_counter.items():
                correct_words += min(count, handwritten_counter.get(word, 0))
            
            word_accuracy = (correct_words / len(printed_words)) * 100
        else:
            word_accuracy = 0
            correct_words = 0
        
        # Calculate punctuation accuracy
        if printed_punct:
            printed_punct_counter = Counter(printed_punct)
            handwritten_punct_counter = Counter(handwritten_punct)
            
            correct_punct = 0
            for punct, count in printed_punct_counter.items():
                correct_punct += min(count, handwritten_punct_counter.get(punct, 0))
            
            punct_accuracy = (correct_punct / len(printed_punct)) * 100
        else:
            punct_accuracy = 100 if len(handwritten_punct) == 0 else 0
            correct_punct = 0
        
        # Overall similarity
        overall_similarity = difflib.SequenceMatcher(None, clean_printed, clean_handwritten).ratio() * 100
        
        return {
            'word_score': f"{correct_words}/{len(printed_words)}",
            'word_accuracy': round(word_accuracy, 1),
            'punct_score': f"{correct_punct}/{len(printed_punct)}",
            'punct_accuracy': round(punct_accuracy, 1),
            'overall_similarity': round(overall_similarity, 1),
            'printed_text': clean_printed,
            'handwritten_text': clean_handwritten,
            'words_printed': len(printed_words),
            'words_handwritten': len(handwritten_words),
            'punct_printed': len(printed_punct),
            'punct_handwritten': len(handwritten_punct)
        }

    def analyze_file_pair(self, file_pair):
        """Analyze a pair of .jpg and .txt files with robust error handling"""
        
        base_name = file_pair['base_name']
        jpg_file = file_pair['jpg_file']
        txt_file = file_pair['txt_file']
        
        print(f"Processing: {base_name}")
        
        try:
            # Extract handwritten text using GPT-4o with retry logic
            print(f"  🤖 Analyzing handwriting with GPT-4o...")
            
            try:
                # Try detailed extraction first
                handwritten_analysis = self.extract_handwritten_text_with_retry(jpg_file, max_retries=3)
            except Exception as e:
                print(f"  ⚠️ Detailed extraction failed, trying simple fallback: {e}")
                try:
                    # Try simple fallback
                    handwritten_analysis = self.extract_handwritten_text_simple_fallback(jpg_file)
                except Exception as e2:
                    print(f"  ❌ All extraction methods failed: {e2}")
                    raise e2
            
            # Extract the text from analysis
            if isinstance(handwritten_analysis, dict):
                handwritten_text = handwritten_analysis.get('extracted_text', '')
                confidence = handwritten_analysis.get('confidence_level', 'unknown')
            else:
                handwritten_text = str(handwritten_analysis)
                confidence = 'unknown'
            
            # Read printed text from file
            printed_text = self.read_printed_text(txt_file)
            
            # Compare the texts
            comparison_results = self.compare_texts(printed_text, handwritten_text)
            
            # Save text comparison to individual file
            self.save_text_comparison(base_name, printed_text, handwritten_analysis, comparison_results)
            
            # Prepare result dictionary
            result = {
                'file_name': base_name,
                'jpg_file': os.path.basename(jpg_file),
                'txt_file': os.path.basename(txt_file),
                'word_score': comparison_results['word_score'],
                'word_accuracy': comparison_results['word_accuracy'],
                'punctuation_score': comparison_results['punct_score'],
                'punctuation_accuracy': comparison_results['punct_accuracy'],
                'overall_similarity': comparison_results['overall_similarity'],
                'tesseract_text': comparison_results['printed_text'],  # For Excel compatibility
                'easyocr_text': comparison_results['handwritten_text'],  # For Excel compatibility
                'gpt4o_confidence': confidence,  # New field for GPT-4o confidence
                'words_count_tesseract': comparison_results['words_printed'],
                'words_count_easyocr': comparison_results['words_handwritten'],
                'punctuation_count_tesseract': comparison_results['punct_printed'],
                'punctuation_count_easyocr': comparison_results['punct_handwritten'],
                'status': 'SUCCESS',
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"  ✓ GPT-4o Confidence: {confidence}")
            print(f"  ✓ Words: {result['word_score']} ({result['word_accuracy']}%)")
            print(f"  ✓ Punctuation: {result['punctuation_score']} ({result['punctuation_accuracy']}%)")
            print(f"  ✓ Overall: {result['overall_similarity']}%")
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            return {
                'file_name': base_name,
                'jpg_file': os.path.basename(jpg_file) if 'jpg_file' in locals() else 'N/A',
                'txt_file': os.path.basename(txt_file) if 'txt_file' in locals() else 'N/A',
                'word_score': 'ERROR',
                'word_accuracy': 0,
                'punctuation_score': 'ERROR',
                'punctuation_accuracy': 0,
                'overall_similarity': 0,
                'tesseract_text': '',
                'easyocr_text': '',
                'gpt4o_confidence': 'error',
                'words_count_tesseract': 0,
                'words_count_easyocr': 0,
                'punctuation_count_tesseract': 0,
                'punctuation_count_easyocr': 0,
                'status': f'ERROR: {str(e)}',
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def process_all_files(self):
        """Process all matching .jpg and .txt file pairs"""
        
        # Find matching file pairs
        file_pairs = self.find_matching_files()
        
        if not file_pairs:
            print(f"No matching .jpg/.txt file pairs found in '{self.data_dir}' directory!")
            return False
        
        print(f"Found {len(file_pairs)} matching .jpg/.txt file pairs to process")
        print(f"Using GPT-4o Vision API with retry logic for reliability")
        print(f"Text comparisons will be saved to: {self.text_output_dir}/")
        print("=" * 60)
        
        # Process each file pair
        for i, file_pair in enumerate(file_pairs, 1):
            print(f"\n[{i}/{len(file_pairs)}] Processing...")
            result = self.analyze_file_pair(file_pair)
            self.results.append(result)
            
            # Add a small delay between requests to avoid rate limiting
            if i < len(file_pairs):  # Don't wait after the last file
                time.sleep(1)  # 1 second delay between requests
        
        print("\n" + "=" * 60)
        print(f"Processing complete! Processed {len(self.results)} file pairs")
        
        return True

    def save_to_excel(self):
        """Save results to Excel file with multiple sheets"""
        
        if not self.results:
            print("No results to save!")
            return False
        
        try:
            # Create DataFrame
            df = pd.DataFrame(self.results)
            
            # Create Excel writer object
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                
                # Main results sheet
                df_main = df[['file_name', 'word_score', 'word_accuracy', 
                             'punctuation_score', 'punctuation_accuracy', 
                             'overall_similarity', 'gpt4o_confidence', 'status']].copy()
                df_main.to_excel(writer, sheet_name='Results Summary', index=False)
                
                # Detailed results sheet
                df_detailed = df[['file_name', 'jpg_file', 'txt_file', 'word_score', 'word_accuracy',
                                'punctuation_score', 'punctuation_accuracy',
                                'overall_similarity', 'gpt4o_confidence', 'words_count_tesseract',
                                'words_count_easyocr', 'punctuation_count_tesseract',
                                'punctuation_count_easyocr', 'processing_time', 'status']].copy()
                df_detailed.to_excel(writer, sheet_name='Detailed Analysis', index=False)
                
                # Text comparison sheet
                df_text = df[['file_name', 'tesseract_text', 'easyocr_text', 'gpt4o_confidence']].copy()
                df_text.columns = ['file_name', 'printed_text', 'handwritten_text_gpt4o', 'gpt4o_confidence']
                df_text.to_excel(writer, sheet_name='Text Comparison', index=False)
                
                # Statistics sheet
                successful = df[df['status'] == 'SUCCESS']
                if not successful.empty:
                    # Count confidence levels
                    confidence_counts = successful['gpt4o_confidence'].value_counts().to_dict()
                    
                    stats = {
                        'Metric': ['Total File Pairs', 'Successful Analyses', 'Failed Analyses',
                                  'Average Word Accuracy', 'Average Punctuation Accuracy',
                                  'Average Overall Similarity', 'Highest Word Accuracy',
                                  'Lowest Word Accuracy', 'Highest Overall Similarity',
                                  'Lowest Overall Similarity', 'GPT-4o High Confidence',
                                  'GPT-4o Medium Confidence', 'GPT-4o Low Confidence'],
                        'Value': [
                            len(df),
                            len(successful),
                            len(df) - len(successful),
                            f"{successful['word_accuracy'].mean():.1f}%",
                            f"{successful['punctuation_accuracy'].mean():.1f}%",
                            f"{successful['overall_similarity'].mean():.1f}%",
                            f"{successful['word_accuracy'].max():.1f}%",
                            f"{successful['word_accuracy'].min():.1f}%",
                            f"{successful['overall_similarity'].max():.1f}%",
                            f"{successful['overall_similarity'].min():.1f}%",
                            confidence_counts.get('high', 0),
                            confidence_counts.get('medium', 0),
                            confidence_counts.get('low', 0)
                        ]
                    }
                    df_stats = pd.DataFrame(stats)
                    df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            print(f"\n✅ Excel results saved to: {self.output_file}")
            print(f"Excel file contains 4 sheets with enhanced error handling results")
            
            return True
            
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            return False

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        
        print("🔬 Biology Handwriting Analysis Pipeline (GPT-4o Robust)")
        print("=" * 60)
        print(f"Data Directory: {self.data_dir}")
        print(f"Excel Output: {self.output_file}")
        print(f"Text Comparisons Directory: {self.text_output_dir}")
        print("OCR Method: GPT-4o Vision API with retry logic and fallbacks")
        print("Comparing: .jpg (handwritten) vs .txt (printed)")
        print()
        
        # Check for OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ Error: OPENAI_API_KEY environment variable not set!")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            return
        
        # Process all file pairs
        if self.process_all_files():
            # Save to Excel
            self.save_to_excel()
            
            # Print summary
            successful = len([r for r in self.results if r['status'] == 'SUCCESS'])
            failed = len(self.results) - successful
            
            print(f"\n📊 ANALYSIS SUMMARY:")
            print(f"Total file pairs processed: {len(self.results)}")
            print(f"Successful analyses: {successful}")
            print(f"Failed analyses: {failed}")
            print(f"Success rate: {(successful/len(self.results)*100):.1f}%")
            
            if successful > 0:
                avg_word_acc = sum(r['word_accuracy'] for r in self.results if r['status'] == 'SUCCESS') / successful
                avg_overall = sum(r['overall_similarity'] for r in self.results if r['status'] == 'SUCCESS') / successful
                print(f"Average word accuracy: {avg_word_acc:.1f}%")
                print(f"Average overall similarity: {avg_overall:.1f}%")
        else:
            print("❌ Analysis failed!")

def main():
    """Main function"""
    
    # Configuration
    data_directory = "Biology-0089-0159"  # Biology directory
    output_filename = "biology_handwriting_analysis_results_gpt4o_robust.xlsx"
    text_output_directory = "biology_text_comparisons_gpt4o_robust"  # Directory for individual text files
    
    # Create analyzer and run
    analyzer = BiologyHandwritingAnalyzerGPT4O(data_directory, output_filename, text_output_directory)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()