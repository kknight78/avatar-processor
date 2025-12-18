#!/usr/bin/env python3
"""
Process avatar photo for HeyGen - HEAD ANCHORED VERSION
1. Remove background via Replicate BiRefNet
2. Detect face to find head position
3. Scale based on HEAD SIZE (not body width)
4. Position head at fixed location
5. Crop anything below frame
"""

import requests
import time
import base64
from PIL import Image
import numpy as np
from io import BytesIO

# Replicate API
import os
REPLICATE_TOKEN = os.environ.get('REPLICATE_TOKEN', '')

# Output dimensions - 16:9 landscape for HeyGen talking head
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
BACKGROUND_COLOR = (128, 128, 128)  # Neutral gray for RVM masking

# Fixed positioning constants (in pixels for 1280x720 frame)
HEAD_TOP_Y = 60              # Top of head (including hat/hair) at exactly 60px from frame top
TARGET_HEAD_HEIGHT = 160     # Target HEAD height (top of head to bottom of chin) in pixels
MIN_SIDE_MARGIN = 0.15       # Minimum 15% margin on sides

def remove_background_birefnet(image_path):
    """Remove background using Replicate's BiRefNet model"""
    print("Removing background with BiRefNet...")

    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    mime = 'image/png' if image_path.lower().endswith('.png') else 'image/jpeg'
    data_uri = f"data:{mime};base64,{image_data}"

    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "version": "95fcc2a26d3899cd6c2691c900465aaeff466285a65c14638cc5f36f34befaf1",
            "input": {"image": data_uri}
        }
    )

    prediction = response.json()
    if 'id' not in prediction:
        raise Exception(f"Failed to create prediction: {prediction}")

    print(f"  Prediction: {prediction['id']}")

    get_url = prediction['urls']['get']
    while True:
        result = requests.get(get_url, headers={"Authorization": f"Bearer {REPLICATE_TOKEN}"}).json()
        if result['status'] == 'succeeded':
            img_response = requests.get(result['output'])
            return Image.open(BytesIO(img_response.content)).convert('RGBA')
        elif result['status'] == 'failed':
            raise Exception(f"BiRefNet failed: {result.get('error')}")
        time.sleep(1)

def detect_face(image_path):
    """Detect face using Replicate face-detection model, returns face bbox"""
    print("Detecting face...")

    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    mime = 'image/png' if image_path.lower().endswith('.png') else 'image/jpeg'
    data_uri = f"data:{mime};base64,{image_data}"

    # Using a simple face detection model
    response = requests.post(
        "https://api.replicate.com/v1/predictions",
        headers={
            "Authorization": f"Bearer {REPLICATE_TOKEN}",
            "Content-Type": "application/json"
        },
        json={
            "version": "8932afc017cd4f63d97693ce6f82de5daff86b54b6deae5629726510ca7ce191",  # marckohlbrugge/face-detect
            "input": {"image": data_uri}
        }
    )

    prediction = response.json()
    if 'id' not in prediction:
        print(f"  Face detection failed, will estimate from person bounds")
        return None

    get_url = prediction['urls']['get']
    while True:
        result = requests.get(get_url, headers={"Authorization": f"Bearer {REPLICATE_TOKEN}"}).json()
        if result['status'] == 'succeeded':
            output = result.get('output')
            print(f"  Raw output: {output}")
            if output:
                # Handle different output formats
                if isinstance(output, list) and len(output) > 0:
                    face = output[0]
                elif isinstance(output, dict):
                    face = output
                else:
                    print("  Unexpected output format, will estimate")
                    return None
                print(f"  Face detected: {face}")
                return face
            print("  No face detected, will estimate")
            return None
        elif result['status'] == 'failed':
            print(f"  Face detection failed, will estimate")
            return None
        time.sleep(1)

def find_person_bounds(img):
    """Find bounding box of non-transparent pixels"""
    arr = np.array(img)
    if arr.shape[2] < 4:
        return 0, 0, img.width, img.height

    alpha = arr[:,:,3]
    non_transparent = alpha > 128

    rows = np.any(non_transparent, axis=1)
    cols = np.any(non_transparent, axis=0)

    if not rows.any() or not cols.any():
        return 0, 0, img.width, img.height

    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])

    return left, top, right, bottom

def estimate_head_from_person(person_top, person_height):
    """Estimate head position as top ~18% of person bounding box"""
    head_top = person_top
    head_height = int(person_height * 0.18)  # Head is roughly 18% of visible body height
    head_bottom = head_top + head_height
    return head_top, head_bottom, head_height

def process_avatar(input_path, output_path):
    """Main processing - anchor to head position"""
    original = Image.open(input_path)
    print(f"Input: {original.width}x{original.height}")

    # Step 1: Remove background
    img_rgba = remove_background_birefnet(input_path)

    # Step 2: Find person bounds
    p_left, p_top, p_right, p_bottom = find_person_bounds(img_rgba)
    person_width = p_right - p_left
    person_height = p_bottom - p_top
    print(f"Person bounds: ({p_left}, {p_top}) to ({p_right}, {p_bottom})")
    print(f"Person size: {person_width}x{person_height}")

    # Step 3: Detect or estimate head position
    face_bbox = detect_face(input_path)

    # Get face bbox for calculating HEAD height (top of head to chin)
    if face_bbox and isinstance(face_bbox, dict) and 'faces' in face_bbox and len(face_bbox['faces']) > 0:
        face = face_bbox['faces'][0]
        face_top = face['y']
        face_height = face['height']
        face_bottom = face_top + face_height
        print(f"Face detected: y={face_top}, height={face_height}, bottom={face_bottom}px")
    else:
        # Fallback: estimate face position
        face_top = p_top + int(person_height * 0.05)
        face_height = int(person_height * 0.15)
        face_bottom = face_top + face_height
        print(f"Face estimated: y={face_top}, height={face_height}, bottom={face_bottom}px")

    # HEAD TOP = top of person bounding box (captures hat/hair)
    head_top = p_top

    # HEAD HEIGHT = from top of head (p_top) to bottom of face (chin)
    # This captures the full head INCLUDING hat/hair
    head_height = face_bottom - head_top
    print(f"Head: top={head_top}px, bottom (chin)={face_bottom}px, height={head_height}px")

    # Step 4: Calculate scale based on HEAD HEIGHT (not face height)
    # This ensures all heads (including hats) are the same size
    scale = TARGET_HEAD_HEIGHT / head_height
    print(f"Scale factor: {scale:.2f} (head: {head_height}px → {TARGET_HEAD_HEIGHT}px)")

    # Safety check: ensure minimum side margins
    scaled_person_width = person_width * scale
    if scaled_person_width > OUTPUT_WIDTH * (1 - 2 * MIN_SIDE_MARGIN):
        max_width = OUTPUT_WIDTH * (1 - 2 * MIN_SIDE_MARGIN)
        scale = max_width / person_width
        print(f"  Adjusted scale for margins: {scale:.2f}")

    # Step 5: Resize the entire image
    new_width = int(img_rgba.width * scale)
    new_height = int(img_rgba.height * scale)
    img_scaled = img_rgba.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Step 6: Calculate position to place HEAD TOP at FIXED Y position
    scaled_head_top = head_top * scale
    y_offset = int(HEAD_TOP_Y - scaled_head_top)

    # Center horizontally based on person center
    scaled_p_left = p_left * scale
    scaled_p_right = p_right * scale
    scaled_person_center_x = (scaled_p_left + scaled_p_right) / 2
    x_offset = int(OUTPUT_WIDTH / 2 - scaled_person_center_x)

    print(f"Placement: x={x_offset}, y={y_offset}")

    # Step 7: Create output canvas and paste
    output = Image.new('RGBA', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (*BACKGROUND_COLOR, 255))
    output.paste(img_scaled, (x_offset, y_offset), img_scaled)

    # Convert to RGB (remove alpha)
    output_rgb = Image.new('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), BACKGROUND_COLOR)
    output_rgb.paste(output, (0, 0), output)

    # Step 8: Save
    output_rgb.save(output_path, 'PNG')

    # Calculate actual margins for reporting
    final_left = x_offset + scaled_p_left
    final_right = OUTPUT_WIDTH - (x_offset + scaled_p_right)
    final_head_top = head_top * scale + y_offset
    print(f"\nResult:")
    print(f"  Head top at: {final_head_top:.0f}px from top (target: {HEAD_TOP_Y}px)")
    print(f"  Head height: {TARGET_HEAD_HEIGHT}px (FIXED - includes hat/hair)")
    print(f"  Side margins: {final_left/OUTPUT_WIDTH*100:.1f}% left, {final_right/OUTPUT_WIDTH*100:.1f}% right")
    print(f"  Saved to: {output_path}")

    return output_rgb

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else '/tmp/avatar-processed.png'
    else:
        input_path = '/Users/kellyknight/Downloads/dad-bg-test.jpg'
        output_path = '/tmp/dad-head-anchored.png'

    process_avatar(input_path, output_path)
