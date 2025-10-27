from paddleocr import PaddleOCR
from PIL import Image
import os
import numpy as np

# Test PaddleOCR v3 with an image
ocr = PaddleOCR(lang='en')

img_path = 'imgs/google_page.png'
if os.path.exists(img_path):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    result = ocr.predict(img_np)
    
    print('Result type:', type(result))
    print('Result[0] type:', type(result[0]))
    print('Keys:', result[0].keys())
    print('\nrec_texts:', result[0]['rec_texts'])
    print('\nrec_scores:', result[0]['rec_scores'])
    print('\nrec_polys shape:', np.array(result[0]['rec_polys']).shape if result[0]['rec_polys'] else 'empty')
    print('\nFirst few polys:', result[0]['rec_polys'][:3] if result[0]['rec_polys'] else 'empty')
else:
    print(f'Image not found: {img_path}')
