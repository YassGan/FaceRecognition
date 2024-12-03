from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import os


### This is just to supress some annoying logs while running the code
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'





def preprocess_image(image_path, output_dir, margin=44, image_size=160):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Initialize MTCNN detector
    detector = MTCNN()

    # Detect faces in the image
    result = detector.detect_faces(img)

    if result:
        for i, face in enumerate(result):
            # Get bounding box and landmarks
            x, y, w, h = face['box']
            keypoints = face['keypoints']
            
            # Add margin around the detected face
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, img.shape[1])
            y2 = min(y + h + margin, img.shape[0])

            # Crop and resize the face
            cropped = img[y1:y2, x1:x2]
            scaled = cv2.resize(cropped, (image_size, image_size))

            # Save the aligned face to the output directory
            output_filename = os.path.join(output_dir, f"{os.path.basename(image_path)}_{i}.jpg")
            cv2.imwrite(output_filename, cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR))  # Save as BGR format

            print(f"Aligned face saved to {output_filename}")
    else:
        print(f"No faces detected in {image_path}")


def align_faces(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate through the dataset directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            image_path = os.path.join(root, file)
            preprocess_image(image_path, output_dir)
            print(f"Processed: {image_path}")

output_dir = "FaceRecognition/data"  # Folder containing your unaligned images
input_dir = "FaceRecognition/aligned_data"  # Folder to save aligned faces

align_faces(input_dir, output_dir)
