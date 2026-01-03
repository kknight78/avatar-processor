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

v16.4-hybrid: Ratio-based sizing with hybrid mode (original background preserved)
"""

import os
import requests
import base64
import time
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# OpenCV DNN Face Detector - downloads model on first use
FACE_DETECTOR = None
def get_face_detector():
    global FACE_DETECTOR
    if FACE_DETECTOR is None:
        # Use OpenCV's built-in Haar Cascade (reliable, no download needed)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        FACE_DETECTOR = cv2.CascadeClassifier(cascade_path)
        print(f"[FACE-DETECT] Loaded Haar Cascade from: {cascade_path}")
    return FACE_DETECTOR

# === RATIO-BASED CONSTANTS (no more fixed pixels!) ===
# These ratios work at ANY resolution

HEAD_TOP_RATIO = 0.375      # Head top positioned at 37.5% from frame top
HEAD_HEIGHT_RATIO = 0.14    # Head (top to chin) should be 14% of frame height
MIN_SIDE_MARGIN_RATIO = 0.05  # Minimum 5% margin on sides

# Output constraints
TARGET_ASPECT_RATIO = 9 / 16  # Portrait 9:16

BACKGROUND_COLOR = (128, 128, 128)  # Neutral gray for RVM masking

# Cloudinary config (optional - for hosted output)
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME', '')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY', '')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET', '')


def detect_face_from_image(img_rgba):
    """
    Detect face using OpenCV Haar Cascade (local, fast, reliable)
    Returns dict with 'y' (face top) and 'height' (face height to chin)
    """
    print(f"[FACE-DETECT] Starting OpenCV face detection, image size: {img_rgba.size}, mode: {img_rgba.mode}")

    try:
        # Convert RGBA to RGB with white background for face detection
        img_rgb = Image.new('RGB', img_rgba.size, (255, 255, 255))
        if img_rgba.mode == 'RGBA':
            img_rgb.paste(img_rgba, mask=img_rgba.split()[3])
        else:
            img_rgb.paste(img_rgba)

        # Convert to numpy array and then to grayscale for OpenCV
        img_array = np.array(img_rgb)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Get face detector
        face_cascade = get_face_detector()

        # Detect faces - try multiple scale factors for reliability
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            # Try with more lenient parameters
            print("[FACE-DETECT] No face with default params, trying lenient...")
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(30, 30)
            )

        if len(faces) == 0:
            print("[FACE-DETECT] OpenCV: No face detected")
            return None

        # Get the largest face (most likely the main subject)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face

        # Haar Cascade detects face box from ~eyebrows to ~jaw line
        # But we need forehead to CHIN for accurate head height
        # The chin is typically ~15-20% below the jaw line detected by Haar
        # Extend the face height to include the chin
        CHIN_EXTENSION = 0.18  # Add 18% to include chin below jaw
        extended_h = int(h * (1 + CHIN_EXTENSION))

        # Convert numpy int32 to Python int for JSON serialization
        face_top = int(y)
        face_height = extended_h

        print(f"[FACE-DETECT] OpenCV found face at: x={x}, y={y}, w={w}, h={h}")
        print(f"[FACE-DETECT] Extended for chin: {h} -> {extended_h} (+{CHIN_EXTENSION*100}%)")
        print(f"[FACE-DETECT] Using face_top={face_top}, face_height={face_height}")

        return {'y': face_top, 'height': face_height}

    except Exception as e:
        print(f"[FACE-DETECT] OpenCV exception: {type(e).__name__}: {e}")
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
        # Always detect from bg-removed image (cleaner background = better detection)
        # even in hybrid mode
        face_data = detect_face_from_image(img_rgba)

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

    # === OUTPUT SIZE derived from head height (NO SCALING!) ===
    # Head should be 14% of output, so output = head / 0.14
    output_height = int(head_height / HEAD_HEIGHT_RATIO)
    output_width = int(output_height * TARGET_ASPECT_RATIO)
    print(f"Output dimensions (from head): {output_width}x{output_height}")

    # Target position for head top
    target_head_top_y = int(output_height * HEAD_TOP_RATIO)
    print(f"Target head position: top at y={target_head_top_y} ({HEAD_TOP_RATIO*100}%)")

    # NO SCALING - just position!
    scale = 1.0
    print(f"Scale factor: {scale:.4f} (NO SCALING - crop/position only)")

    # Calculate position to place HEAD TOP at target Y position
    # head_top is position in input image, we want it at target_head_top_y in output
    y_offset = int(target_head_top_y - head_top)

    # Center horizontally based on person center
    person_center_x = (p_left + p_right) / 2
    x_offset = int(output_width / 2 - person_center_x)

    print(f"Positioning: x_offset={x_offset}, y_offset={y_offset}")

    # Create output canvas - NO SCALING, just crop/position
    if use_original:
        # HYBRID MODE: Gray background only at edges, original photo with its background
        output_rgb = Image.new('RGB', (output_width, output_height), BACKGROUND_COLOR)
        # Convert original to RGB if needed
        img_to_paste = output_source_img
        if img_to_paste.mode == 'RGBA':
            img_rgb = Image.new('RGB', img_to_paste.size, BACKGROUND_COLOR)
            img_rgb.paste(img_to_paste, (0, 0), img_to_paste)
            img_to_paste = img_rgb
        elif img_to_paste.mode != 'RGB':
            img_to_paste = img_to_paste.convert('RGB')
        # Paste original photo (with its background) onto gray canvas - NO SCALING
        output_rgb.paste(img_to_paste, (x_offset, y_offset))
        print("[PROCESS] Output: Original photo positioned (NO SCALING), gray padding at edges")
    else:
        # LEGACY MODE: Full gray background with transparent person
        output = Image.new('RGBA', (output_width, output_height), (*BACKGROUND_COLOR, 255))
        output.paste(output_source_img, (x_offset, y_offset), output_source_img)
        output_rgb = Image.new('RGB', (output_width, output_height), BACKGROUND_COLOR)
        output_rgb.paste(output, (0, 0), output)
        print("[PROCESS] Output: BG-removed person positioned (NO SCALING), gray background")

    # Check if legs were cropped (person extends beyond canvas)
    person_bottom_in_output = p_bottom + y_offset
    legs_cropped = person_bottom_in_output > output_height

    return output_rgb, {
        'version': 'v19-no-scale',
        'mode': 'hybrid_original' if use_original else 'bg_removed_only',
        'input_size': f'{img_rgba.width}x{img_rgba.height}',
        'original_size': f'{original_img.width}x{original_img.height}' if use_original else None,
        'output_size': f'{output_width}x{output_height}',
        'scale': 1.0,
        'head_height': head_height,
        'head_percent_of_output': round(head_height / output_height * 100, 1),
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
        'version': 'v19-no-scale',
        'approach': 'NO SCALING - crop and position only',
        'head_top_ratio': HEAD_TOP_RATIO,
        'head_height_ratio': HEAD_HEIGHT_RATIO,
        'face_detection': 'opencv_haar_cascade (local, reliable)'
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

        # Run MediaPipe face detection
        face_data = detect_face_from_image(img)

        return jsonify({
            'success': True,
            'image_size': f'{img.width}x{img.height}',
            'face_data': face_data,
            'detection_method': 'mediapipe'
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
    print(f"Starting Avatar Processor API v19-no-scale on port {port}")
    print(f"Approach: NO SCALING - crop and position only")
    print(f"Face detection: OpenCV Haar Cascade (local, reliable)")
    print(f"Head top ratio: {HEAD_TOP_RATIO} ({HEAD_TOP_RATIO*100}%)")
    print(f"Head height ratio: {HEAD_HEIGHT_RATIO} ({HEAD_HEIGHT_RATIO*100}%)")
    app.run(host='0.0.0.0', port=port, debug=True)
