#!/usr/bin/env python3
"""
Avatar Photo Processor API
Flask API that processes avatar photos for HeyGen:
1. Downloads background-removed image
2. Detects face via Replicate to find chin position
3. Scales based on head height (top of head to chin)
4. Positions head at fixed ratio from top
5. Output size derived from face quality (not fixed resolution)
6. Returns processed image URL (via Cloudinary or base64)

v15: Ratio-based sizing - preserves face quality, output size driven by input face resolution
"""

import os
import requests
import base64
import time
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__)

# Replicate API token for face detection
REPLICATE_TOKEN = os.environ.get('REPLICATE_TOKEN', '')

# === RATIO-BASED CONSTANTS (no more fixed pixels!) ===
# These ratios work at ANY resolution

HEAD_TOP_RATIO = 0.375      # Head top positioned at 37.5% from frame top
HEAD_HEIGHT_RATIO = 0.14    # Head (top to chin) should be 14% of frame height
MIN_SIDE_MARGIN_RATIO = 0.05  # Minimum 5% margin on sides

# Output constraints
TARGET_ASPECT_RATIO = 9 / 16  # Portrait 9:16
MAX_OUTPUT_HEIGHT = 3840      # Cap at 4K to avoid crazy file sizes
MIN_OUTPUT_HEIGHT = 1280      # Minimum for reasonable quality

BACKGROUND_COLOR = (128, 128, 128)  # Neutral gray for RVM masking

# Cloudinary config (optional - for hosted output)
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME', '')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY', '')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET', '')


def detect_face_from_image(img_rgba):
    """
    Detect face using Replicate face-detection model
    Returns face bbox dict with 'y' and 'height' for chin position
    """
    print(f"[FACE-DETECT] Starting face detection, image size: {img_rgba.size}, mode: {img_rgba.mode}")
    print(f"[FACE-DETECT] REPLICATE_TOKEN present: {bool(REPLICATE_TOKEN)}, length: {len(REPLICATE_TOKEN) if REPLICATE_TOKEN else 0}")

    if not REPLICATE_TOKEN:
        print("[FACE-DETECT] No REPLICATE_TOKEN, skipping face detection")
        return None

    # Convert image to base64 for Replicate
    buffered = BytesIO()
    # Convert RGBA to RGB for face detection
    img_rgb = Image.new('RGB', img_rgba.size, (255, 255, 255))
    img_rgb.paste(img_rgba, mask=img_rgba.split()[3] if img_rgba.mode == 'RGBA' else None)
    img_rgb.save(buffered, format='PNG')
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data_uri = f"data:image/png;base64,{image_data}"

    print(f"[FACE-DETECT] Base64 data length: {len(image_data)}")

    try:
        # Using marckohlbrugge/face-detect model
        print("[FACE-DETECT] Sending prediction request to Replicate...")
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Bearer {REPLICATE_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "version": "8932afc017cd4f63d97693ce6f82de5daff86b54b6deae5629726510ca7ce191",
                "input": {"image": data_uri}
            },
            timeout=30
        )

        print(f"[FACE-DETECT] Prediction response status: {response.status_code}")

        prediction = response.json()
        if 'id' not in prediction:
            print(f"[FACE-DETECT] Failed to start - response: {prediction}")
            return None

        print(f"[FACE-DETECT] Prediction started: {prediction.get('id')}")

        get_url = prediction['urls']['get']

        # Poll for result (max 30 seconds)
        for i in range(30):
            result = requests.get(
                get_url,
                headers={"Authorization": f"Bearer {REPLICATE_TOKEN}"},
                timeout=10
            ).json()

            status = result['status']
            if i % 5 == 0:  # Log every 5 seconds
                print(f"[FACE-DETECT] Poll {i}: status={status}")

            if status == 'succeeded':
                output = result.get('output')
                print(f"[FACE-DETECT] Success! Raw output: {output}")

                if output:
                    # Handle different output formats from the model
                    if isinstance(output, dict) and 'faces' in output and len(output['faces']) > 0:
                        face = output['faces'][0]
                        print(f"[FACE-DETECT] Found face: y={face['y']}, height={face['height']}")
                        return {'y': face['y'], 'height': face['height']}
                    elif isinstance(output, list) and len(output) > 0:
                        face = output[0]
                        if 'y' in face and 'height' in face:
                            print(f"[FACE-DETECT] Found face (list): y={face['y']}, height={face['height']}")
                            return {'y': face['y'], 'height': face['height']}
                    elif isinstance(output, dict) and 'y' in output:
                        print(f"[FACE-DETECT] Found face (dict): y={output['y']}, height={output['height']}")
                        return {'y': output['y'], 'height': output['height']}

                print(f"[FACE-DETECT] No face found in output structure: {type(output)}")
                return None

            elif status == 'failed':
                print(f"[FACE-DETECT] Prediction failed: {result.get('error')}")
                return None

            time.sleep(1)

        print("[FACE-DETECT] Timed out after 30 seconds")
        return None

    except Exception as e:
        print(f"[FACE-DETECT] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_person_bounds(img):
    """Find bounding box of non-transparent pixels"""
    arr = np.array(img)
    if len(arr.shape) < 3 or arr.shape[2] < 4:
        return 0, 0, img.width, img.height

    alpha = arr[:, :, 3]
    non_transparent = alpha > 128

    rows = np.any(non_transparent, axis=1)
    cols = np.any(non_transparent, axis=0)

    if not rows.any() or not cols.any():
        return 0, 0, img.width, img.height

    top = int(np.argmax(rows))
    bottom = int(len(rows) - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - np.argmax(cols[::-1]))

    return left, top, right, bottom


def calculate_output_dimensions(head_height_pixels):
    """
    Calculate output dimensions based on face quality.

    The head height in the INPUT determines the output size.
    This preserves face quality - bigger face = bigger output.

    Args:
        head_height_pixels: Height of head (top to chin) in input image pixels

    Returns:
        (output_width, output_height) tuple
    """
    # Head should be HEAD_HEIGHT_RATIO (14%) of output height
    # So: output_height = head_height_pixels / HEAD_HEIGHT_RATIO
    output_height = int(head_height_pixels / HEAD_HEIGHT_RATIO)

    # Clamp to reasonable bounds
    output_height = max(MIN_OUTPUT_HEIGHT, min(MAX_OUTPUT_HEIGHT, output_height))

    # Width from 9:16 aspect ratio
    output_width = int(output_height * TARGET_ASPECT_RATIO)

    return output_width, output_height


def process_avatar_image(img_rgba, face_data=None, original_img=None):
    """
    Process avatar image: scale and position based on head size.
    Output size is determined by face quality (not fixed resolution).

    Args:
        img_rgba: PIL Image with transparency (RGBA) - used for bounds detection
        face_data: dict with face detection results {'y', 'height'}
        original_img: PIL Image - original photo with background (optional)
                      If provided, output uses original with its background intact.
                      If not provided, output uses bg-removed image with gray background.

    Returns:
        PIL Image (RGB), processing_info dict
    """
    # Determine which image to use for output
    use_original = original_img is not None
    output_source_img = original_img if use_original else img_rgba

    print(f"[PROCESS] Mode: {'HYBRID (original + bg-removed for bounds)' if use_original else 'BG-REMOVED only'}")

    # Find person bounds from bg-removed image (accurate head top detection)
    p_left, p_top, p_right, p_bottom = find_person_bounds(img_rgba)
    person_width = p_right - p_left
    person_height = p_bottom - p_top

    print(f"Input image: {img_rgba.width}x{img_rgba.height}")
    if use_original:
        print(f"Original image: {original_img.width}x{original_img.height}")
    print(f"Person bounds (from transparency): left={p_left}, top={p_top}, right={p_right}, bottom={p_bottom}")
    print(f"Person size: {person_width}x{person_height}")

    # If no face_data provided, detect face automatically
    if not face_data or 'y' not in face_data:
        print("No face_data provided, running face detection...")
        # Detect from original if available (better quality), else from bg-removed
        detect_source = original_img if use_original else img_rgba
        face_data = detect_face_from_image(detect_source)

    # Calculate head position from face detection
    face_detected = False
    if face_data and 'y' in face_data and 'height' in face_data:
        face_top = face_data['y']
        face_height = face_data['height']
        face_bottom = face_top + face_height
        face_detected = True
        print(f"Using detected face: y={face_top}, height={face_height}, chin at {face_bottom}")
    else:
        # Fallback: estimate face position (only if face detection fails)
        print("Face detection failed, falling back to estimation")
        face_top = p_top + int(person_height * 0.05)
        face_height = int(person_height * 0.15)
        face_bottom = face_top + face_height

    # HEAD TOP = top of person bounding box (captures hat/hair) - from transparency
    head_top = p_top

    # HEAD HEIGHT = from top of head to bottom of face (chin)
    head_height = face_bottom - head_top

    if head_height <= 0:
        head_height = int(person_height * 0.2)
        face_detected = False

    print(f"Head height (top to chin): {head_height}px")

    # === Output size based on face quality ===
    output_width, output_height = calculate_output_dimensions(head_height)
    print(f"Output dimensions (based on face quality): {output_width}x{output_height}")

    # Calculate target values in pixels for this output size
    target_head_top_y = int(output_height * HEAD_TOP_RATIO)
    target_head_height = int(output_height * HEAD_HEIGHT_RATIO)

    print(f"Target head position: top at y={target_head_top_y} ({HEAD_TOP_RATIO*100}%)")
    print(f"Target head height: {target_head_height}px ({HEAD_HEIGHT_RATIO*100}%)")

    # Calculate scale to achieve target head height
    scale = target_head_height / head_height
    print(f"Scale factor: {scale:.4f}")

    # Safety check: ensure minimum side margins
    scaled_person_width = person_width * scale
    max_allowed_width = output_width * (1 - 2 * MIN_SIDE_MARGIN_RATIO)
    if scaled_person_width > max_allowed_width:
        old_scale = scale
        scale = max_allowed_width / person_width
        print(f"Adjusted scale for side margins: {old_scale:.4f} -> {scale:.4f}")

    # Resize the source image (original or bg-removed)
    new_width = int(output_source_img.width * scale)
    new_height = int(output_source_img.height * scale)
    img_scaled = output_source_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Scaled image: {new_width}x{new_height}")

    # Calculate position to place HEAD TOP at target Y position
    scaled_head_top = head_top * scale
    y_offset = int(target_head_top_y - scaled_head_top)

    # Center horizontally based on person center (from transparency bounds)
    scaled_p_left = p_left * scale
    scaled_p_right = p_right * scale
    scaled_person_center_x = (scaled_p_left + scaled_p_right) / 2
    x_offset = int(output_width / 2 - scaled_person_center_x)

    print(f"Positioning: x_offset={x_offset}, y_offset={y_offset}")

    # Create output canvas
    if use_original:
        # HYBRID MODE: Gray background only at edges, original photo with its background
        output_rgb = Image.new('RGB', (output_width, output_height), BACKGROUND_COLOR)
        # Convert original to RGB if needed
        if img_scaled.mode == 'RGBA':
            img_scaled_rgb = Image.new('RGB', img_scaled.size, BACKGROUND_COLOR)
            img_scaled_rgb.paste(img_scaled, (0, 0), img_scaled)
            img_scaled = img_scaled_rgb
        elif img_scaled.mode != 'RGB':
            img_scaled = img_scaled.convert('RGB')
        # Paste original photo (with its background) onto gray canvas
        output_rgb.paste(img_scaled, (x_offset, y_offset))
        print("[PROCESS] Output: Original photo with background, gray padding at edges")
    else:
        # LEGACY MODE: Full gray background with transparent person
        output = Image.new('RGBA', (output_width, output_height), (*BACKGROUND_COLOR, 255))
        output.paste(img_scaled, (x_offset, y_offset), img_scaled)
        output_rgb = Image.new('RGB', (output_width, output_height), BACKGROUND_COLOR)
        output_rgb.paste(output, (0, 0), output)
        print("[PROCESS] Output: BG-removed person on full gray background")

    # Check if legs were cropped
    scaled_person_bottom = p_bottom * scale + y_offset
    legs_cropped = scaled_person_bottom > output_height

    return output_rgb, {
        'version': 'v16-hybrid',
        'mode': 'hybrid_original' if use_original else 'bg_removed_only',
        'input_size': f'{img_rgba.width}x{img_rgba.height}',
        'original_size': f'{original_img.width}x{original_img.height}' if use_original else None,
        'output_size': f'{output_width}x{output_height}',
        'scale': round(scale, 4),
        'head_height_input': head_height,
        'head_height_output': int(head_height * scale),
        'target_head_height': target_head_height,
        'face_detected': face_detected,
        'legs_cropped': legs_cropped,
        'x_offset': x_offset,
        'y_offset': y_offset,
        'person_bounds': {
            'left': p_left,
            'top': p_top,
            'right': p_right,
            'bottom': p_bottom
        }
    }


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'version': 'v16-hybrid',
        'approach': 'Output size driven by face quality',
        'head_top_ratio': HEAD_TOP_RATIO,
        'head_height_ratio': HEAD_HEIGHT_RATIO,
        'max_output_height': MAX_OUTPUT_HEIGHT,
        'min_output_height': MIN_OUTPUT_HEIGHT,
        'face_detection': 'enabled' if REPLICATE_TOKEN else 'disabled',
        'replicate_token_length': len(REPLICATE_TOKEN) if REPLICATE_TOKEN else 0
    })


@app.route('/test-face-detect', methods=['POST'])
def test_face_detect():
    """
    Test face detection on an image URL
    Request JSON: {"image_url": "..."}
    """
    try:
        data = request.get_json()
        if not data or 'image_url' not in data:
            return jsonify({'error': 'Missing image_url'}), 400

        image_url = data['image_url']

        # Download image
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to download: {response.status_code}'}), 400

        img = Image.open(BytesIO(response.content)).convert('RGBA')

        # Run face detection
        face_data = detect_face_from_image(img)

        return jsonify({
            'success': True,
            'image_size': f'{img.width}x{img.height}',
            'face_data': face_data,
            'replicate_token_present': bool(REPLICATE_TOKEN),
            'replicate_token_length': len(REPLICATE_TOKEN) if REPLICATE_TOKEN else 0
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/process', methods=['POST'])
def process():
    """
    Process an avatar image (v16 - Hybrid Mode)

    Request JSON:
        - image_url: URL of background-removed image (PNG with transparency)
        - original_image_url: (optional) URL of original image WITH background
                              If provided, enables HYBRID MODE:
                              - Uses bg-removed for accurate head bounds
                              - Outputs original photo with its background intact
                              - Gray padding only at canvas edges
        - face_data: (optional) Face detection result {'y', 'height'}
        - output_format: 'base64' or 'url' (default: 'base64')

    Response JSON:
        - success: boolean
        - processed_image: base64 string or URL
        - processing_info: dict with scale/position details
        - dimensions: {width, height} of output
    """
    try:
        data = request.get_json()

        if not data or 'image_url' not in data:
            return jsonify({'success': False, 'error': 'Missing image_url'}), 400

        image_url = data['image_url']
        original_image_url = data.get('original_image_url')
        original_image_base64 = data.get('original_image_base64')  # Alternative to URL
        face_data = data.get('face_data')
        output_format = data.get('output_format', 'base64')

        # Determine if we have an original image source
        has_original = original_image_url or original_image_base64

        print(f"\n{'='*60}")
        print(f"Processing avatar image (v16)")
        print(f"BG-removed URL: {image_url[:100]}...")
        if original_image_url:
            print(f"Original URL: {original_image_url[:100]}...")
        if original_image_base64:
            print(f"Original base64: [{len(original_image_base64)} chars]")
        print(f"MODE: {'HYBRID (original with background)' if has_original else 'LEGACY (bg-removed only)'}")
        print(f"{'='*60}")

        # Download the bg-removed image (always needed for bounds detection)
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to download bg-removed image: {response.status_code}'}), 400

        # Open as RGBA
        img_bg_removed = Image.open(BytesIO(response.content)).convert('RGBA')
        print(f"Downloaded bg-removed image: {img_bg_removed.width}x{img_bg_removed.height}")

        # Get original image if provided (for hybrid mode)
        original_img = None
        if original_image_base64:
            # Original provided as base64
            print(f"Loading original image from base64 for hybrid mode...")
            try:
                # Handle data URI format
                if original_image_base64.startswith('data:'):
                    original_image_base64 = original_image_base64.split(',', 1)[1]
                img_data = base64.b64decode(original_image_base64)
                original_img = Image.open(BytesIO(img_data)).convert('RGBA')
                print(f"Loaded original image: {original_img.width}x{original_img.height}")
            except Exception as e:
                print(f"Error loading original from base64: {e}, falling back to legacy mode")
        elif original_image_url:
            # Original provided as URL
            print(f"Downloading original image for hybrid mode...")
            try:
                orig_response = requests.get(original_image_url, timeout=30)
                if orig_response.status_code == 200:
                    original_img = Image.open(BytesIO(orig_response.content)).convert('RGBA')
                    print(f"Downloaded original image: {original_img.width}x{original_img.height}")
                else:
                    print(f"Failed to download original: {orig_response.status_code}, falling back to legacy mode")
            except Exception as e:
                print(f"Error downloading original image: {e}, falling back to legacy mode")

        # Process the image (hybrid mode if original available)
        processed_img, processing_info = process_avatar_image(
            img_rgba=img_bg_removed,
            face_data=face_data,
            original_img=original_img
        )

        print(f"\nProcessing complete!")
        print(f"Mode: {processing_info['mode']}")
        print(f"Output: {processing_info['output_size']}")
        print(f"Scale: {processing_info['scale']}")
        print(f"Face detected: {processing_info['face_detected']}")
        print(f"Legs cropped: {processing_info['legs_cropped']}")

        # Output
        if output_format == 'base64':
            buffered = BytesIO()
            processed_img.save(buffered, format='PNG', optimize=False)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({
                'success': True,
                'processed_image': f'data:image/png;base64,{img_base64}',
                'processing_info': processing_info,
                'dimensions': {
                    'width': processed_img.width,
                    'height': processed_img.height
                }
            })
        else:
            # For URL output, would need Cloudinary upload
            return jsonify({'success': False, 'error': 'URL output not yet implemented'}), 501

    except Exception as e:
        import traceback
        print(f"Error processing avatar: {e}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Avatar Processor API v15 (ratio-based) on port {port}")
    print(f"Head top ratio: {HEAD_TOP_RATIO} ({HEAD_TOP_RATIO*100}%)")
    print(f"Head height ratio: {HEAD_HEIGHT_RATIO} ({HEAD_HEIGHT_RATIO*100}%)")
    print(f"Output height range: {MIN_OUTPUT_HEIGHT} - {MAX_OUTPUT_HEIGHT}")
    app.run(host='0.0.0.0', port=port, debug=True)
