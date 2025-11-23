import cv2
import pytesseract
import re
import numpy as np
import os

class MedicalOCR:
    def __init__(self, tesseract_cmd_path=None):
        # Set tesseract path if provided (useful for Windows users)
        # e.g., r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

    def preprocess_image(self, image_path):
        """
        Reads an image and applies preprocessing to improve OCR accuracy.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Otsu's thresholding (binarization)
        # This creates a high-contrast black and white image
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        # Noise reduction (Denoising)
        gray = cv2.medianBlur(gray, 3)
        
        return gray

    def extract_text(self, processed_img):
        """Runs Tesseract OCR on the preprocessed image."""
        # Configuration: --psm 6 assumes a single uniform block of text
        custom_config = r'--oem 3 --psm 6' 
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        return text

    def parse_medical_data(self, raw_text):
        """
        Extracts structured data (BP, Cholesterol, etc.) from raw text
        using Regular Expressions.
        """
        data = {}
        
        # Regex patterns to find values even if text is messy
        patterns = {
            'cholesterol': r'(?:cholesterol|chol|total chol)\D*(\d{2,3})',
            'bp': r'(?:bp|blood pressure)\D*(\d{2,3}\s*/\s*\d{2,3})',
            'age': r'(?:age|years)\D*(\d{1,3})',
            'glucose': r'(?:glucose|sugar|bs)\D*(\d{2,3})'
        }
        
        for key, pattern in patterns.items():
            # Search for pattern, case insensitive
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                data[key] = match.group(1)
            else:
                data[key] = None
                
        return data

    def run_pipeline(self, image_path):
        print(f"--- Processing Report: {image_path} ---")
        
        # 1. Preprocess
        processed_img = self.preprocess_image(image_path)
        
        # 2. OCR Extraction
        raw_text = self.extract_text(processed_img)
        print(f"[Raw Text Extracted]:\n{raw_text[:200]}...") # Print first 200 chars
        
        # 3. Parsing
        structured_data = self.parse_medical_data(raw_text)
        print(f"[Structured Data]: {structured_data}")
        
        return structured_data