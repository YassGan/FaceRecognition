import os
import cv2
from mtcnn.mtcnn import MTCNN


def preprocess_image(input_dir, output_dir, margin=44, image_size=160):
    # Check if output directory exists, create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through each person's folder in the input directory
    for person_folder in os.listdir(input_dir):
        person_folder_path = os.path.join(input_dir, person_folder)

        # Check if it's a directory (person folder)
        if os.path.isdir(person_folder_path):
            # Create the same folder structure in the output directory
            output_person_folder = os.path.join(output_dir, person_folder)
            if not os.path.exists(output_person_folder):
                os.makedirs(output_person_folder)

            # Loop through each image in the person's folder
            for image_name in os.listdir(person_folder_path):
                image_path = os.path.join(person_folder_path, image_name)

                # Process the image
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
                        output_filename = os.path.join(output_person_folder, f"{os.path.splitext(image_name)[0]}_{i}.jpg")
                        cv2.imwrite(output_filename, cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR))  # Save as BGR format

                        print(f"Aligned face saved to {output_filename}")
                else:
                    print(f"No faces detected in {image_path}")
        else:
            print(f"Skipping non-directory item: {person_folder}")


# Example usage
input_dir = "FaceRecognition/data"
output_dir = "FaceRecognition/aligned_data"
preprocess_image(input_dir, output_dir)



preprocess_image(input_dir, output_dir)