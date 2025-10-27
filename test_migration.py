"""Test script to verify PaddleOCR v3 migration"""
import sys
import numpy as np
from PIL import Image
from util.utils import check_ocr_box

print("Testing PaddleOCR v3 migration...")
print("=" * 60)

# Test with a real image
image_path = 'imgs/google_page.png'
print(f"\nTest 1: Loading image from {image_path}")

try:
    image = Image.open(image_path)
    print(f"Image size: {image.size}")
    
    # Test with use_paddleocr=True
    print("\nTest 2: Running OCR with PaddleOCR v3 (use_paddleocr=True)")
    (text, ocr_bbox), _ = check_ocr_box(
        image, 
        display_img=False, 
        output_bb_format='xyxy', 
        easyocr_args={'text_threshold': 0.9}, 
        use_paddleocr=True
    )
    
    print(f"Number of detected text boxes: {len(text)}")
    print(f"Number of bounding boxes: {len(ocr_bbox)}")
    
    if len(text) > 0:
        print(f"\nFirst 5 detected texts:")
        for i, txt in enumerate(text[:5]):
            print(f"  {i}: {txt}")
        
        print(f"\nFirst 3 bounding boxes (xyxy format):")
        for i, bbox in enumerate(ocr_bbox[:3]):
            print(f"  {i}: {bbox}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] PaddleOCR v3 migration test PASSED!")
    
except Exception as e:
    print(f"\n[ERROR] Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
