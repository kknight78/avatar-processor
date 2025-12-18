#!/usr/bin/env python3
"""
Avatar Photo Processor API
Flask API that processes avatar photos for HeyGen:
1. Downloads background-removed image
2. Detects face via Replicate to find chin position
3. Scales based on head height (top of head to chin)
4. Positions head at fixed location
5. Returns processed image URL (via Cloudinary or base64)
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

# Output dimensions - 9:16 portrait (TikTok/Reels standard)
OUTPUT_WIDTH = 720
OUTPUT_HEIGHT = 1280
BACKGROUND_COLOR = (128, 128, 128)  # Neutral gray for RVM masking

# Fixed positioning constants (in pixels for 720x1280 frame)
HEAD_TOP_Y = 480             # Top of head at 480px from frame top
TARGET_HEAD_HEIGHT = 180     # Target head height in pixels (smaller for more arm room)
MIN_SIDE_MARGIN = 0.05       # Minimum 5% margin on sides

# Cloudinary config (optional - for hosted output)
CLOUDINARY_CLOUD_NAME = os.environ.get('CLOUDINARY_CLOUD_NAME', '')
CLOUDINARY_API_KEY = os.environ.get('CLOUDINARY_API_KEY', '')
CLOUDINARY_API_SECRET = os.environ.get('CLOUDINARY_API_SECRET', '')


def detect_face_from_image(img_rgba):
    """
    Detect face using Replicate face-detection model
    Returns face bbox dict with 'y' and 'height' for chin position
    """
    if not REPLICATE_TOKEN:
        print("No REPLICATE_TOKEN, skipping face detection")
        return None

    # Convert image to base64 for Replicate
    buffered = BytesIO()
    # Convert RGBA to RGB for face detection
    img_rgb = Image.new('RGB', img_rgba.size, (255, 255, 255))
    img_rgb.paste(img_rgba, mask=img_rgba.split()[3] if img_rgba.mode == 'RGBA' else None)
    img_rgb.save(buffered, format='PNG')
    image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data_uri = f"data:image/png;base64,{image_data}"

    try:
        # Using marckohlbrugge/face-detect model
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

        prediction = response.json()
        if 'id' not in prediction:
            print(f"Face detection failed to start: {prediction}")
            return None

        get_url = prediction['urls']['get']

        # Poll for result (max 30 seconds)
        for _ in range(30):
            result = requests.get(
                get_url,
                headers={"Authorization": f"Bearer {REPLICATE_TOKEN}"},
                timeout=10
            ).json()

            if result['status'] == 'succeeded':
                output = result.get('output')
                print(f"Face detection raw output: {output}")

                if output:
                    # Handle different output formats from the model
                    if isinstance(output, dict) and 'faces' in output and len(output['faces']) > 0:
                        face = output['faces'][0]
                        return {'y': face['y'], 'height': face['height']}
                    elif isinstance(output, list) and len(output) > 0:
                        face = output[0]
                        if 'y' in face and 'height' in face:
                            return {'y': face['y'], 'height': face['height']}
                    elif isinstance(output, dict) and 'y' in output:
                        return {'y': output['y'], 'height': output['height']}

                print("No face found in output")
                return None

            elif result['status'] == 'failed':
                print(f"Face detection failed: {result.get('error')}")
                return None

            time.sleep(1)

        print("Face detection timed out")
        return None

    except Exception as e:
        print(f"Face detection error: {e}")
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


def process_avatar_image(img_rgba, face_data=None):
    """
    Process avatar image: scale and position based on head size

    Args:
        img_rgba: PIL Image with transparency (RGBA)
        face_data: dict with face detection results {'y', 'height'}

    Returns:
        PIL Image (RGB) with gray background
    """
    # Find person bounds
    p_left, p_top, p_right, p_bottom = find_person_bounds(img_rgba)
    person_width = p_right - p_left
    person_height = p_bottom - p_top

    # If no face_data provided, detect face automatically
    if not face_data or 'y' not in face_data:
        print("No face_data provided, running face detection...")
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

    # HEAD TOP = top of person bounding box (captures hat/hair)
    head_top = p_top

    # HEAD HEIGHT = from top of head to bottom of face (chin)
    head_height = face_bottom - head_top

    if head_height <= 0:
        head_height = int(person_height * 0.2)
        face_detected = False

    # Calculate scale based on HEAD HEIGHT
    scale = TARGET_HEAD_HEIGHT / head_height

    # Safety check: ensure minimum side margins
    scaled_person_width = person_width * scale
    if scaled_person_width > OUTPUT_WIDTH * (1 - 2 * MIN_SIDE_MARGIN):
        max_width = OUTPUT_WIDTH * (1 - 2 * MIN_SIDE_MARGIN)
        scale = max_width / person_width

    # Resize the entire image
    new_width = int(img_rgba.width * scale)
    new_height = int(img_rgba.height * scale)
    img_scaled = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Calculate position to place HEAD TOP at FIXED Y position
    scaled_head_top = head_top * scale
    y_offset = int(HEAD_TOP_Y - scaled_head_top)

    # Center horizontally based on person center
    scaled_p_left = p_left * scale
    scaled_p_right = p_right * scale
    scaled_person_center_x = (scaled_p_left + scaled_p_right) / 2
    x_offset = int(OUTPUT_WIDTH / 2 - scaled_person_center_x)

    # Create output canvas and paste
    output = Image.new('RGBA', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (*BACKGROUND_COLOR, 255))
    output.paste(img_scaled, (x_offset, y_offset), img_scaled)

    # Convert to RGB (remove alpha)
    output_rgb = Image.new('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), BACKGROUND_COLOR)
    output_rgb.paste(output, (0, 0), output)

    return output_rgb, {
        'scale': scale,
        'head_height_original': head_height,
        'head_height_final': TARGET_HEAD_HEIGHT,
        'face_detected': face_detected,
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
        'version': '14-face-detection',
        'output_size': f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}',
        'head_top_y': HEAD_TOP_Y,
        'target_head_height': TARGET_HEAD_HEIGHT,
        'face_detection': 'enabled' if REPLICATE_TOKEN else 'disabled'
    })


@app.route('/process', methods=['POST'])
def process():
    """
    Process an avatar image

    Request JSON:
        - image_url: URL of background-removed image (PNG with transparency)
        - original_image_url: (optional) URL of original image for face detection
        - face_data: (optional) Face detection result {'y', 'height'}
        - output_format: 'base64' or 'url' (default: 'base64')

    Response JSON:
        - success: boolean
        - processed_image: base64 string or URL
        - processing_info: dict with scale/position details
    """
    try:
        data = request.get_json()

        if not data or 'image_url' not in data:
            return jsonify({'success': False, 'error': 'Missing image_url'}), 400

        image_url = data['image_url']
        original_image_url = data.get('original_image_url')
        face_data = data.get('face_data')
        output_format = data.get('output_format', 'base64')

        # Download the bg-removed image
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            return jsonify({'success': False, 'error': f'Failed to download image: {response.status_code}'}), 400

        # Open as RGBA
        img = Image.open(BytesIO(response.content)).convert('RGBA')

        # If original_image_url provided and no face_data, detect face from original
        if original_image_url and not face_data:
            print(f"Detecting face from original image: {original_image_url}")
            try:
                orig_response = requests.get(original_image_url, timeout=30)
                if orig_response.status_code == 200:
                    orig_img = Image.open(BytesIO(orig_response.content)).convert('RGBA')
                    face_data = detect_face_from_image(orig_img)
                    if face_data:
                        print(f"Face detected from original: {face_data}")
                    else:
                        print("No face detected from original, will try bg-removed image")
            except Exception as e:
                print(f"Error downloading original image: {e}")

        # Process the image
        processed_img, processing_info = process_avatar_image(img, face_data)

        # Output
        if output_format == 'base64':
            buffered = BytesIO()
            processed_img.save(buffered, format='PNG')
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return jsonify({
                'success': True,
                'processed_image': f'data:image/png;base64,{img_base64}',
                'processing_info': processing_info,
                'dimensions': {
                    'width': OUTPUT_WIDTH,
                    'height': OUTPUT_HEIGHT
                }
            })
        else:
            # For URL output, would need Cloudinary upload
            return jsonify({'success': False, 'error': 'URL output not yet implemented'}), 501

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
