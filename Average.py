import os
from tqdm import tqdm
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


# Function to generate embeddings from a folder of face images
def generate_embeddings_from_folder(folder_path):
    embeddings = {}
    face_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('png', 'jpg', 'jpeg'))]
    for face_path in tqdm(face_paths):
        try:
            embedding = DeepFace.represent(img_path=face_path, model_name='Facenet')[0]['embedding']
            relative_path = os.path.relpath(face_path, start=folder_path)  # Make it relative
            embeddings[relative_path] = embedding
        except Exception as e:
            print(f"Error processing {face_path}: {e}")
    return embeddings


# Store the influencer's embeddings and performance tracking
unique_faces = {}
influencer_performance = {}
influencer_videos = {}

# Path to the pre-extracted faces folder
faces_folder_path = "faces" 

# Generate embeddings directly from the pre-extracted images
embeddings = generate_embeddings_from_folder(faces_folder_path)

# Simulate performance data (you would replace this with real data if available)
# Here we assume performance values are associated with each image for processing purposes
performance_data = {face_path: np.random.rand() for face_path in embeddings.keys()}  # Example performance values

# Process embeddings and calculate influencer tracking logic
for face_path, embedding in embeddings.items():
    assigned = False
    for influencer_id, ref_embedding in unique_faces.items():
        if cosine_similarity([embedding], [ref_embedding])[0][0] > 0.8:  # Threshold for similarity
            # Associate this face with the influencer
            influencer_performance[influencer_id].append(performance_data[face_path])
            influencer_videos[influencer_id].append(face_path)
            assigned = True
            break

    if not assigned:
        # Assign a new influencer ID
        new_influencer_id = f"influencer_{len(unique_faces) + 1}"
        unique_faces[new_influencer_id] = embedding
        influencer_performance[new_influencer_id] = [performance_data[face_path]]
        influencer_videos[new_influencer_id] = [face_path]

# Calculate the average performance for each influencer
average_performance = {influencer_id: np.mean(performance)
                       for influencer_id, performance in influencer_performance.items()}

# Prepare the output DataFrame to include video associations
output_data = []

for influencer_id in average_performance:
    videos = ", ".join(influencer_videos[influencer_id])  # Join video URLs
    avg_performance = average_performance[influencer_id]
    output_data.append([influencer_id, avg_performance, videos])

# Save the output to CSV
output_df = pd.DataFrame(output_data, columns=["Influencer ID", "Average Performance", "Face Paths"])

output_df.to_csv("influencer_performance_with_faces.csv", index=False)
print("Output saved as influencer_performance_with_faces.csv")

# Display the output DataFrame
print(output_df.head())
