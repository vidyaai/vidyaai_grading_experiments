import cv2
import pytesseract
import easyocr
import re
import difflib
from collections import Counter
import os
import pandas as pd
from datetime import datetime
import glob
from pathlib import Path
import string

class HandwritingAnalyzer:
    def __init__(self, data_directory="handwriting_data_new", output_file="handwriting_analysis_results_improved.xlsx", text_output_dir="text_comparisons_improved"):
        self.data_dir = data_directory
        self.output_file = output_file
        self.text_output_dir = text_output_dir
        self.results = []
        
        # Create text output directory if it doesn't exist
        os.makedirs(self.text_output_dir, exist_ok=True)
        
        # Initialize EasyOCR reader once for efficiency
        print("Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(['en'])
        print("EasyOCR reader initialized successfully!")

    def preprocess_image(self, image_path):
        """Improve image quality for better OCR results"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply image preprocessing to reduce noise
        # 1. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # 2. Threshold to get binary image
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return gray, cleaned

    def is_valid_text_chunk(self, text_chunk):
        """Check if a text chunk is likely to be real text"""
        if not text_chunk or len(text_chunk.strip()) < 2:
            return False
        
        # Remove whitespace for analysis
        clean_chunk = text_chunk.strip()
        
        # Check ratio of letters to other characters
        letter_count = sum(1 for c in clean_chunk if c.isalpha())
        total_count = len(clean_chunk)
        
        if total_count == 0:
            return False
        
        letter_ratio = letter_count / total_count
        
        # Accept if at least 50% of characters are letters
        # OR if it contains common punctuation/numbers with letters
        if letter_ratio >= 0.5:
            return True
        
        # Accept if it's mostly punctuation but contains some letters
        if letter_count > 0 and any(c in string.punctuation for c in clean_chunk):
            return True
        
        # Reject chunks that are mostly symbols or random characters
        return False

    def extract_text_from_image(self, image_path):
        """Extract text with improved preprocessing and filtering"""
        
        # Preprocess image
        gray, cleaned = self.preprocess_image(image_path)
        
        # Method 1: Tesseract with safe configuration (no problematic quotes)
        # Using simple config to avoid quote errors
        custom_config = r'--oem 3 --psm 6'
        tesseract_text = pytesseract.image_to_string(cleaned, config=custom_config)
        
        # Method 2: EasyOCR with higher confidence threshold
        easyocr_results = self.reader.readtext(gray)
        # Use higher confidence threshold and filter by text quality
        easyocr_text = ' '.join([
            result[1] for result in easyocr_results 
            if result[2] > 0.6 and self.is_valid_text_chunk(result[1])  # Higher confidence + validation
        ])
        
        return tesseract_text, easyocr_text

    def is_coherent_sentence(self, sentence):
        """Check if a sentence is coherent (not OCR garbage)"""
        if not sentence or len(sentence.strip()) < 5:
            return False
        
        words = sentence.split()
        if len(words) < 2:
            return False
        
        # Check if most words look like real words
        valid_words = 0
        for word in words:
            # Remove punctuation for word validation
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) >= 2 and clean_word.isalpha():
                valid_words += 1
        
        # Consider coherent if at least 60% of words look valid
        return (valid_words / len(words)) >= 0.6

    def clean_text(self, text):
        """Advanced text cleaning to remove OCR artifacts"""
        if not text:
            return ""
        
        # Step 1: Basic normalization
        text = text.strip()
        
        # Step 2: Remove obvious OCR artifacts
        # Remove sequences of random characters (likely OCR errors)
        text = re.sub(r'\b[a-z]{1,2}\b(?:\s+[a-z]{1,2}\b){3,}', ' ', text)  # Remove sequences of 1-2 letter "words"
        
        # Remove isolated numbers/symbols that don't make sense
        text = re.sub(r'\b\d+[a-z~!@#$%^&*()_+={}[\]|\\:";\'<>?,./]*\b', ' ', text)
        
        # Remove sequences of special characters
        text = re.sub(r'[~!@#$%^&*()_+={}[\]|\\:";\'<>?,./]{3,}', ' ', text)
        
        # Remove standalone special characters (except common punctuation)
        text = re.sub(r'\s[^a-zA-Z0-9.,!?;:()\[\]{}"\'\\/-]\s', ' ', text)
        
        # Step 3: Clean up common OCR misreadings
        # Fix common OCR substitutions
        ocr_fixes = {
            r'\bl\b': 'I',  # lowercase l mistaken for I
            r'\b0\b': 'O',  # zero mistaken for O in some contexts
            r'rn': 'm',     # rn often misread as m
            r'\bvv\b': 'w', # double v misread as w
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text)
        
        # Step 4: Remove trailing garbage
        # Split into sentences and keep only the coherent parts
        sentences = re.split(r'[.!?]+', text)
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and self.is_coherent_sentence(sentence):
                clean_sentences.append(sentence)
            else:
                # Stop processing when we hit incoherent text (likely artifacts)
                break
        
        # Rejoin sentences
        if clean_sentences:
            text = '. '.join(clean_sentences)
            if not text.endswith('.'):
                text += '.'
        
        # Step 5: Final cleanup
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        return text

    def save_text_comparison(self, filename, tesseract_text_raw, easyocr_text_raw, tesseract_text_clean, easyocr_text_clean, comparison_results):
        """Save text comparison including raw and cleaned versions"""
        
        # Create filename without extension and add .txt
        base_name = os.path.splitext(filename)[0]
        text_filename = f"{base_name}.txt"
        text_filepath = os.path.join(self.text_output_dir, text_filename)
        
        try:
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"TEXT COMPARISON FOR: {filename}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Raw extracted texts
                f.write("RAW EXTRACTED TEXTS (Before Cleaning):\n")
                f.write("-" * 50 + "\n")
                f.write("TESSERACT OCR (Raw Output):\n")
                f.write(f'"""{tesseract_text_raw[:500]}..."""\n\n')  # Limit length for readability
                
                f.write("EASYOCR (Raw Output):\n")
                f.write(f'"""{easyocr_text_raw[:500]}..."""\n\n')
                
                # Cleaned texts
                f.write("CLEANED & FILTERED TEXTS (After Processing):\n")
                f.write("-" * 50 + "\n")
                f.write("Cleaned Tesseract Text:\n")
                f.write(f'"{tesseract_text_clean}"\n\n')
                
                f.write("Cleaned EasyOCR Text:\n")
                f.write(f'"{easyocr_text_clean}"\n\n')
                
                # Analysis results
                f.write("ANALYSIS RESULTS:\n")
                f.write("-" * 50 + "\n")
                f.write(f"Word Score: {comparison_results['word_score']} ({comparison_results['word_accuracy']}% accuracy)\n")
                f.write(f"Punctuation Score: {comparison_results['punct_score']} ({comparison_results['punct_accuracy']}% accuracy)\n")
                f.write(f"Overall Similarity: {comparison_results['overall_similarity']}%\n\n")
                
                # Word breakdown
                f.write("WORD ANALYSIS:\n")
                f.write("-" * 50 + "\n")
                tesseract_words = re.findall(r'\w+', tesseract_text_clean)
                easyocr_words = re.findall(r'\w+', easyocr_text_clean)
                
                f.write(f"Tesseract detected words ({len(tesseract_words)}): {tesseract_words}\n")
                f.write(f"EasyOCR detected words ({len(easyocr_words)}): {easyocr_words}\n\n")
                
                # Word matching analysis
                if tesseract_words and easyocr_words:
                    f.write("WORD MATCHING ANALYSIS:\n")
                    tesseract_counter = Counter(tesseract_words)
                    easyocr_counter = Counter(easyocr_words)
                    
                    f.write("Matched words:\n")
                    matched_words = []
                    for word in tesseract_counter:
                        if word in easyocr_counter:
                            matched_words.append(word)
                            f.write(f"  - '{word}'\n")
                    
                    f.write("Words only in Tesseract:\n")
                    for word in tesseract_counter:
                        if word not in easyocr_counter:
                            f.write(f"  - '{word}'\n")
                    
                    f.write("Words only in EasyOCR:\n")
                    for word in easyocr_counter:
                        if word not in tesseract_counter:
                            f.write(f"  - '{word}'\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("END OF COMPARISON\n")
                f.write("=" * 80 + "\n")
                
            print(f"  Text comparison saved: {text_filepath}")
            return True
            
        except Exception as e:
            print(f"  Error saving text comparison: {e}")
            return False

    def compare_texts(self, text1, text2):
        """Compare two texts and return similarity scores"""
        
        # Clean texts
        clean_text1 = self.clean_text(text1)
        clean_text2 = self.clean_text(text2)
        
        # Extract words
        words1 = re.findall(r'\w+', clean_text1)
        words2 = re.findall(r'\w+', clean_text2)
        
        # Extract punctuation
        punct1 = re.findall(r'[^\w\s]', clean_text1)
        punct2 = re.findall(r'[^\w\s]', clean_text2)
        
        # Calculate word accuracy
        if words1:
            matcher = difflib.SequenceMatcher(None, words1, words2)
            word_similarity = matcher.ratio()
            
            # Count exact word matches
            words1_counter = Counter(words1)
            words2_counter = Counter(words2)
            
            correct_words = 0
            for word, count in words1_counter.items():
                correct_words += min(count, words2_counter.get(word, 0))
            
            word_accuracy = (correct_words / len(words1)) * 100
        else:
            word_accuracy = 0
            correct_words = 0
        
        # Calculate punctuation accuracy
        if punct1:
            punct1_counter = Counter(punct1)
            punct2_counter = Counter(punct2)
            
            correct_punct = 0
            for punct, count in punct1_counter.items():
                correct_punct += min(count, punct2_counter.get(punct, 0))
            
            punct_accuracy = (correct_punct / len(punct1)) * 100
        else:
            punct_accuracy = 100 if len(punct2) == 0 else 0
            correct_punct = 0
        
        # Overall similarity
        overall_similarity = difflib.SequenceMatcher(None, clean_text1, clean_text2).ratio() * 100
        
        return {
            'word_score': f"{correct_words}/{len(words1)}",
            'word_accuracy': round(word_accuracy, 1),
            'punct_score': f"{correct_punct}/{len(punct1)}",
            'punct_accuracy': round(punct_accuracy, 1),
            'overall_similarity': round(overall_similarity, 1),
            'tesseract_text': clean_text1,
            'easyocr_text': clean_text2,
            'words_tesseract': len(words1),
            'words_easyocr': len(words2),
            'punct_tesseract': len(punct1),
            'punct_easyocr': len(punct2)
        }

    def analyze_image(self, image_path):
        """Analyze a single image and return results"""
        
        filename = os.path.basename(image_path)
        print(f"Processing: {filename}")
        
        try:
            # Extract text using both methods
            tesseract_text_raw, easyocr_text_raw = self.extract_text_from_image(image_path)
            
            # Clean the texts
            tesseract_text_clean = self.clean_text(tesseract_text_raw)
            easyocr_text_clean = self.clean_text(easyocr_text_raw)
            
            # Compare the cleaned texts
            comparison_results = self.compare_texts(tesseract_text_raw, easyocr_text_raw)
            
            # Save text comparison including raw and cleaned versions
            self.save_text_comparison(
                filename, 
                tesseract_text_raw, 
                easyocr_text_raw,
                tesseract_text_clean, 
                easyocr_text_clean, 
                comparison_results
            )
            
            # Prepare result dictionary
            result = {
                'file_name': filename,
                'file_path': image_path,
                'word_score': comparison_results['word_score'],
                'word_accuracy': comparison_results['word_accuracy'],
                'punctuation_score': comparison_results['punct_score'],
                'punctuation_accuracy': comparison_results['punct_accuracy'],
                'overall_similarity': comparison_results['overall_similarity'],
                'tesseract_text': comparison_results['tesseract_text'],
                'easyocr_text': comparison_results['easyocr_text'],
                'words_count_tesseract': comparison_results['words_tesseract'],
                'words_count_easyocr': comparison_results['words_easyocr'],
                'punctuation_count_tesseract': comparison_results['punct_tesseract'],
                'punctuation_count_easyocr': comparison_results['punct_easyocr'],
                'status': 'SUCCESS',
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"  Words: {result['word_score']} ({result['word_accuracy']}%)")
            print(f"  Punctuation: {result['punctuation_score']} ({result['punctuation_accuracy']}%)")
            print(f"  Overall: {result['overall_similarity']}%")
            
            return result
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            return {
                'file_name': filename,
                'file_path': image_path,
                'word_score': 'ERROR',
                'word_accuracy': 0,
                'punctuation_score': 'ERROR',
                'punctuation_accuracy': 0,
                'overall_similarity': 0,
                'tesseract_text': '',
                'easyocr_text': '',
                'words_count_tesseract': 0,
                'words_count_easyocr': 0,
                'punctuation_count_tesseract': 0,
                'punctuation_count_easyocr': 0,
                'status': f'ERROR: {str(e)}',
                'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def process_all_images(self):
        """Process all PNG images in the data directory"""
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            print(f"Error: Directory '{self.data_dir}' not found!")
            return False
        
        # Find all PNG files
        png_pattern = os.path.join(self.data_dir, "*.png")
        image_files = glob.glob(png_pattern)
        
        if not image_files:
            print(f"No PNG files found in '{self.data_dir}' directory!")
            return False
        
        print(f"Found {len(image_files)} PNG files to process")
        print(f"Text comparisons will be saved to: {self.text_output_dir}/")
        print("=" * 60)
        
        # Process each image
        for i, image_path in enumerate(sorted(image_files), 1):
            print(f"\n[{i}/{len(image_files)}] Processing...")
            result = self.analyze_image(image_path)
            self.results.append(result)
        
        print("\n" + "=" * 60)
        print(f"Processing complete! Processed {len(self.results)} files")
        
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
                             'overall_similarity', 'status']].copy()
                df_main.to_excel(writer, sheet_name='Results Summary', index=False)
                
                # Detailed results sheet
                df_detailed = df[['file_name', 'word_score', 'word_accuracy',
                                'punctuation_score', 'punctuation_accuracy',
                                'overall_similarity', 'words_count_tesseract',
                                'words_count_easyocr', 'punctuation_count_tesseract',
                                'punctuation_count_easyocr', 'processing_time', 'status']].copy()
                df_detailed.to_excel(writer, sheet_name='Detailed Analysis', index=False)
                
                # Text comparison sheet
                df_text = df[['file_name', 'tesseract_text', 'easyocr_text']].copy()
                df_text.to_excel(writer, sheet_name='Text Comparison', index=False)
                
                # Statistics sheet
                successful = df[df['status'] == 'SUCCESS']
                if not successful.empty:
                    stats = {
                        'Metric': ['Total Files', 'Successful Analyses', 'Failed Analyses',
                                  'Average Word Accuracy', 'Average Punctuation Accuracy',
                                  'Average Overall Similarity', 'Highest Word Accuracy',
                                  'Lowest Word Accuracy', 'Highest Overall Similarity',
                                  'Lowest Overall Similarity'],
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
                            f"{successful['overall_similarity'].min():.1f}%"
                        ]
                    }
                    df_stats = pd.DataFrame(stats)
                    df_stats.to_excel(writer, sheet_name='Statistics', index=False)
            
            print(f"\nExcel results saved to: {self.output_file}")
            print(f"Excel file contains 4 sheets:")
            print(f"  - Results Summary: Main results")
            print(f"  - Detailed Analysis: Complete analysis data")
            print(f"  - Text Comparison: OCR text outputs")
            print(f"  - Statistics: Summary statistics")
            
            return True
            
        except Exception as e:
            print(f"Error saving to Excel: {e}")
            return False

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        
        print("Handwriting Analysis Pipeline - IMPROVED VERSION")
        print("=" * 60)
        print(f"Data Directory: {self.data_dir}")
        print(f"Excel Output: {self.output_file}")
        print(f"Text Comparisons Directory: {self.text_output_dir}")
        print()
        print("IMPROVEMENTS:")
        print("- Higher EasyOCR confidence threshold (0.6 vs 0.3)")
        print("- Advanced OCR artifact removal")
        print("- Image preprocessing for better OCR")
        print("- Text validation to filter garbage")
        print("- Fixed Tesseract configuration issues")
        print()
        
        # Process all images
        if self.process_all_images():
            # Save to Excel
            self.save_to_excel()
            
            # Print summary
            successful = len([r for r in self.results if r['status'] == 'SUCCESS'])
            failed = len(self.results) - successful
            
            print(f"\nANALYSIS SUMMARY:")
            print(f"Total files processed: {len(self.results)}")
            print(f"Successful analyses: {successful}")
            print(f"Failed analyses: {failed}")
            print(f"Individual text files created: {successful}")
            print(f"Text files location: {self.text_output_dir}/")
            
            if successful > 0:
                avg_word_acc = sum(r['word_accuracy'] for r in self.results if r['status'] == 'SUCCESS') / successful
                avg_overall = sum(r['overall_similarity'] for r in self.results if r['status'] == 'SUCCESS') / successful
                print(f"Average word accuracy: {avg_word_acc:.1f}%")
                print(f"Average overall similarity: {avg_overall:.1f}%")
                
                print(f"\nCHECK YOUR RESULTS:")
                print(f"- Excel file: {self.output_file}")
                print(f"- Individual text comparisons: {self.text_output_dir}/")
        else:
            print("Analysis failed!")

def main():
    """Main function"""
    
    # Configuration
    data_directory = "handwriting_data_new"  # Change this if your directory has a different name
    output_filename = "handwriting_analysis_results_improved.xlsx"
    text_output_directory = "text_comparisons_improved"  # Directory for individual text files
    
    # Create analyzer and run
    analyzer = HandwritingAnalyzer(data_directory, output_filename, text_output_directory)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()