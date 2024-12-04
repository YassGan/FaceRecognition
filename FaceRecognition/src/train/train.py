import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import random

# Initialize MTCNN and InceptionResnetV1 (FaceNet)
mtcnn = MTCNN(keep_all=True)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Step 1: Simple function for a single image
def extract_embedding(image_path, mtcnn, facenet_model):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces using MTCNN
    faces = mtcnn(img_rgb)
    if faces is None:
        print(f"No faces detected in {image_path}")
        return None
    
    # Use the first detected face (if multiple faces are detected, modify if needed)
    face = faces[0]
    
    # Get embedding for the face
    embedding = facenet_model(face.unsqueeze(0))
    return embedding.detach().numpy()  # Return as numpy array

# Step 2: Function to process all images in a directory
def extract_embeddings_from_directory(image_dir, mtcnn, facenet_model):
    embeddings = []
    labels = []
    
    for person_name in os.listdir(image_dir):
        person_dir = os.path.join(image_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            embedding = extract_embedding(image_path, mtcnn, facenet_model)
            
            if embedding is not None:
                embeddings.append(embedding)
                labels.append(person_name)
    
    return np.array(embeddings), np.array(labels)

# Example usage
aligned_images_input_dir = "FaceRecognition/aligned_data"  # Modify as needed
embeddings, labels = extract_embeddings_from_directory(aligned_images_input_dir, mtcnn, facenet_model)

# Print the shape of the embeddings to check if the extraction was successful
print(f"Embeddings shape: {embeddings.shape}")
print(f"Labels: {labels}")


# Étape 2 : Créer des triplets (Anchor, Positive, Negative)
def create_triplets(embeddings, labels, num_triplets=100):
    triplets = []
    
    for _ in range(num_triplets):
        # Sélectionner un anchor
        anchor_idx = random.randint(0, len(embeddings) - 1)
        anchor_embedding = embeddings[anchor_idx]
        anchor_label = labels[anchor_idx]
        
        # Sélectionner un positif (même étiquette que l'ancre)
        positive_idx = random.choice([i for i, label in enumerate(labels) if label == anchor_label and i != anchor_idx])
        positive_embedding = embeddings[positive_idx]
        
        # Sélectionner un négatif (étiquette différente)
        negative_idx = random.choice([i for i, label in enumerate(labels) if label != anchor_label])
        negative_embedding = embeddings[negative_idx]
        
        triplets.append([anchor_embedding, positive_embedding, negative_embedding])

    return np.array(triplets)


num_triplets = 100
triplets = create_triplets(embeddings, labels, num_triplets)





# # Étape 3 : Définir la fonction de perte triplet
# def triplet_loss(y_true, y_pred, alpha=0.2):
#     """
#     Fonction de perte triplet. Encourage l'ancre à se rapprocher du positif et à s'éloigner du négatif.
#     """
#     anchor, positive, negative = tf.split(y_pred, 3, axis=-1)
#     pos_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
#     neg_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
#     loss = tf.maximum(pos_distance - neg_distance + alpha, 0.0)
#     return tf.reduce_mean(loss)

# # Étape 4 : Créer le modèle triplet
# def create_triplet_model(base_model):
#     input_anchor = layers.Input(shape=(160, 160, 3), name='anchor')
#     input_positive = layers.Input(shape=(160, 160, 3), name='positive')
#     input_negative = layers.Input(shape=(160, 160, 3), name='negative')
    
#     anchor_embedding = base_model(input_anchor)
#     positive_embedding = base_model(input_positive)
#     negative_embedding = base_model(input_negative)
    
#     embeddings = layers.concatenate([anchor_embedding, positive_embedding, negative_embedding], axis=-1)
#     triplet_model = models.Model(inputs=[input_anchor, input_positive, input_negative], outputs=embeddings)
#     triplet_model.compile(optimizer='adam', loss=triplet_loss)
    
#     return triplet_model

# # Exemple d'utilisation
# aligned_images_input_dir = "FaceRecognition/aligned_data"

# # Extraire les embeddings des visages alignés
# embeddings, labels = extract_embeddings(aligned_images_input_dir)

# # Créer des triplets (anchor, positive, negative)

# # Définir le modèle FaceNet de base
# base_model = InceptionResnetV1(pretrained='vggface2').eval()

# # Créer et entraîner le modèle avec perte triplet
# triplet_model = create_triplet_model(base_model)
# triplet_model.fit([triplets[:, 0], triplets[:, 1], triplets[:, 2]], epochs=10)
