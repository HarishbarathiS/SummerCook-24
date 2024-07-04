import cv2
import torch
from torchvision import transforms
from face_embedding import SiameseNetwork, recognize_face, preprocess_data

# Haar cascade classifer to detect facial pixels
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def real_time_face_recognition(embeddings, labels):
    # capture webcam
    cap = cv2.VideoCapture(0)  
    
    while True:
        # read frames from webcam
        ret, frame = cap.read()  
        if not ret:
            break
        # convert frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if boxes is not None:
            # for each face detected
            for box in boxes:
                # (x, y) -> top-right corner co-ordinates ; (w, h) -> width and height
                x, y, w, h = box.astype(int)
                
                # maintain the bounding box is within the frame
                x, y = max(0, x), max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                
                # extract face region and preprocess
                if w > 0 and h > 0:
                    face_region = frame[y:y+h, x:x+w]
                    if face_region.size > 0:
                        face_resized = cv2.resize(face_region, (160, 160))
                        face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
                        
                        # generate embedding using Siamese network
                        siamese_net = SiameseNetwork()
                        embedding = siamese_net(face_tensor).squeeze()
                        
                        # recognize the face using the embedding
                        recognized_label = recognize_face(embedding, embeddings, labels)
                        
                        # draw bounding box and label on the frame
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, recognized_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        
        # display the frame with recognized faces
        cv2.imshow('Real-time Face Recognition', frame)
        
        # break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # directory containing face images
    dir = "."
    
    # preprocess data 
    embeddings, labels = preprocess_data(dir)
    
    print(f"Loaded {len(embeddings)} faces.")
    
    real_time_face_recognition(embeddings, labels)