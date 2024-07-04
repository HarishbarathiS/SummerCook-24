import torch
import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import numpy as np


# # MTCNN module for face detection
# mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# haar cascade is a pre-trained model for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_data(directory):
    # store embeddings and corresponding labels
    embeddings = []
    labels = []
    # load the images from the file system 
    for person_name in os.listdir(directory):
        person_dir = os.path.join(directory, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                # encode images to numerical values 
                image = cv2.imread(image_path)
                if image is not None:
                    embedding = extract_face_embedding(image, person_name)
                    if embedding is not None:
                        embeddings.append(embedding)
                        labels.append(person_name)
    
    return embeddings, labels


# class for face recognition
class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # embedding model
        self.embedding_net = InceptionResnetV1(pretrained='vggface2').eval() 

    def forward(self, x):
        return self.embedding_net(x)


def extract_face_embedding(image,name):
    # # Convert image to RGB and 
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # boxes, _ = mtcnn.detect(rgb_image)

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box.astype(int)
        
            # maintain bounding box is within the image
            x, y = max(0, x), max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w > 0 and h > 0:
                # extract face region and preprocess
                face_region = image[y:y+h, x:x+w]
                if face_region.size > 0:
                    # resize to match model's expected input size
                    face_resized = cv2.resize(face_region, (160, 160)) 
                    # convert image to tensor values 
                    face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
                    
                    # generate embedding using Siamese network
                    siamese_net = SiameseNetwork()
                    embedding = siamese_net(face_tensor).squeeze()
                    
                    return embedding
    
    return None

def recognize_face(embedding, known_embeddings, known_labels):
    if not known_embeddings:
        return "Unknown"
    
    # calculates euclidean distance with stored embeddings
    distances = [torch.norm(embedding - known_emb) for known_emb in known_embeddings]
     
    threshold = 1.0
    
    # find the closest match based on the minimum distance
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    
    # label of the closest match
    recognized_label = known_labels[min_index]
    
    # determine if the face is recognized based on the threshold
    if min_distance < threshold:
        return recognized_label
    else:
        return "Unknown"
    

if __name__ == "__main__":
    # directory containing face images
    dir = "."
    
    # data preprocessing
    embeddings, labels = preprocess_data(dir)
    