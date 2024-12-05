import cv2
import ollama
import os
import sys

def extract_frames(video_path):
    # Ensure video exists
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        sys.exit(1)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        sys.exit(1)

    # Read the first frame
    success, first_frame = cap.read()
    if success:
        first_frame_path = "first_frame.png"
        cv2.imwrite(first_frame_path, first_frame)
    else:
        print("Error: Could not read the first frame.")
        sys.exit(1)

    # Move to the last frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)

    # Read the last frame
    success, last_frame = cap.read()
    if success:
        last_frame_path = "last_frame.png"
        cv2.imwrite(last_frame_path, last_frame)
    else:
        print("Error: Could not read the last frame.")
        sys.exit(1)

    cap.release()
    return first_frame_path, last_frame_path

def analyze_image(image_path):
    try:
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': 'What is in this image?',
                'images': [image_path]
            }]
        )
        ##print(f"Response for {image_path}: {response}")
        return response
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <video_file_path>")
        sys.exit(1)

    video_file = sys.argv[1]

    # Extract frames
    first_frame, last_frame = extract_frames(video_file)

    # Analyze the first frame
    d1=analyze_image(first_frame)

    # Analyze the last frame
    d2=analyze_image(last_frame)
    print(d1['message']['content'])
    print(d2['message']['content'])
