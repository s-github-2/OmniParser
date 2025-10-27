# PaddleOCR v2.7.3 to v3.x Migration Notes

## Summary
Successfully migrated OmniParser codebase from PaddleOCR v2.7.3 to v3.x (specifically v3.3.0).

## Changes Made

### 1. Updated `requirements.txt`
- Changed `paddlepaddle` to `paddlepaddle>=3.0.0`
- Changed `paddleocr` to `paddleocr>=3.0.0`

### 2. Updated `util/utils.py`

#### PaddleOCR Initialization (Lines 20-32)
**Key Changes:**
- Removed `use_angle_cls` → Use `use_textline_orientation` instead
- Removed `use_gpu` → Use `device='cpu'` or `device='gpu'` instead
- Removed `show_log` → Not supported in v3 constructor
- Removed `det`, `rec`, `cls` → These are auto-enabled in v3
- Removed `use_dilation` and `det_db_score_mode` → May need to be set via other parameters
- Changed `rec_batch_num` → `text_recognition_batch_size`

**Before:**
```python
paddle_ocr = PaddleOCR(
    lang='en',
    use_angle_cls=False,
    use_gpu=False,
    show_log=False,
    use_dilation=True,
    det_db_score_mode='slow',
    rec_batch_num=1024
)
```

**After:**
```python
paddle_ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=False,
    device='cpu',
    text_recognition_batch_size=1024
)
```

#### OCR Method Call and Result Parsing (Lines 512-535)
**Key Changes:**
- Changed `ocr()` → `predict()` (ocr() is deprecated)
- Removed `cls=False` parameter from method call
- Completely new result format

**Old Result Format (v2.7.3):**
```python
result = paddle_ocr.ocr(image_np, cls=False)[0]
# result = [
#   (polygon, (text, score)),
#   ...
# ]
coord = [item[0] for item in result if item[1][1] > text_threshold]
text = [item[1][0] for item in result if item[1][1] > text_threshold]
```

**New Result Format (v3.x):**
```python
result_list = paddle_ocr.predict(image_np)
ocr_result = result_list[0]  # OCRResult object (dict-like)
# ocr_result has:
#   - rec_polys: list of polygons (numpy arrays, shape (4,2))
#   - rec_texts: list of recognized texts
#   - rec_scores: list of confidence scores

rec_polys = ocr_result['rec_polys']
rec_texts = ocr_result['rec_texts']
rec_scores = ocr_result['rec_scores']

coord = []
text = []
for poly, txt, score in zip(rec_polys, rec_texts, rec_scores):
    if score > text_threshold:
        coord.append(poly.tolist())
        text.append(txt)
```

## API Changes Reference

### Constructor Parameters
| v2.7.3 Parameter | v3.x Replacement | Notes |
|------------------|------------------|-------|
| `use_angle_cls` | `use_textline_orientation` | Renamed |
| `use_gpu` | `device='cpu'/'gpu'` | Changed to device string |
| `show_log` | N/A | Not available in constructor |
| `det=True/False` | Auto-enabled | Models auto-enabled based on pipeline |
| `rec=True/False` | Auto-enabled | Models auto-enabled based on pipeline |
| `cls=True/False` | Auto-enabled | Models auto-enabled based on pipeline |
| `rec_batch_num` | `text_recognition_batch_size` | Renamed |
| `use_dilation` | N/A | May be available via other parameters |
| `det_db_score_mode` | N/A | May be available via other parameters |

### Method Calls
| v2.7.3 | v3.x | Notes |
|--------|------|-------|
| `ocr(img, cls=False)` | `predict(img)` | `ocr()` deprecated, no cls param |

### Result Structure
| v2.7.3 | v3.x | Notes |
|--------|------|-------|
| `result[0]` returns list of `(polygon, (text, score))` | `result[0]` returns OCRResult object | Complete restructure |
| Access: `result[0][i][0]` for polygon | Access: `result[0]['rec_polys'][i]` | Dict-like access |
| Access: `result[0][i][1][0]` for text | Access: `result[0]['rec_texts'][i]` | Simpler access |
| Access: `result[0][i][1][1]` for score | Access: `result[0]['rec_scores'][i]` | Simpler access |
| Polygon is nested list | Polygon is numpy array (4, 2) | Need `.tolist()` conversion |

## Testing

Run the migration test:
```bash
python test_migration.py
```

Expected output:
- Detects text boxes from test image
- Shows recognized texts and bounding boxes
- Prints "[SUCCESS] PaddleOCR v3 migration test PASSED!"

## Known Issues
- Some advanced parameters like `use_dilation` and `det_db_score_mode` may not have direct equivalents in v3
- Model files are auto-downloaded to `~/.paddlex/official_models/` on first run
- The v3 API includes additional preprocessing steps (doc orientation, unwarping) which may affect performance

## Compatibility
- Tested with PaddleOCR 3.3.0
- Tested with PaddlePaddle 3.2.0
- Python 3.12
- Windows 10/11

## Files Modified
1. `requirements.txt` - Updated version constraints
2. `util/utils.py` - Updated PaddleOCR initialization and result parsing
3. `test_migration.py` - New test script to verify migration
4. `MIGRATION_NOTES.md` - This documentation

## No Changes Required
- `gradio_demo.py` - No changes needed (uses `check_ocr_box` function)
- `util/omniparser.py` - No changes needed (uses `check_ocr_box` function)
- `demo.ipynb` - No changes needed (uses `check_ocr_box` function)
