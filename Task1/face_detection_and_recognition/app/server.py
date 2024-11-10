from flask import Flask, request, jsonify, render_template
from face_embedding import preprocess_data, recognize_face_for_image, extract_face_embedding, SiameseNetwork
import cv2
from torchvision import transforms
import base64
import torch
from PIL import Image
import io
import numpy as np

app = Flask(__name__,  template_folder="../templates")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load embeddings and initialize model
embeddings_dict = preprocess_data(r'D:/HARISH/AspireNex-Tasks/Task1/face_detection_and_recognition')  # Path to stored face images
siamese_net = SiameseNetwork().to('cuda' if torch.cuda.is_available() else 'cpu')

# def real_time_face_recognition(embeddings):
#     cap = cv2.VideoCapture(0)
#     frame_skip = 1  # Process every 2nd frame to reduce lag
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        

#         # Resize frame for faster processing
#         scale_factor = 0.5
#         small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
#         gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
#         if frame_count % frame_skip == 0:
#             # Detect faces in the downscaled grayscale frame
#             boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
#             for box in boxes:
#                 x, y, w, h = [int(coord / scale_factor) for coord in box]  # Scale back to original frame size
                
#                 # Extract and preprocess the face region
#                 face_region = frame[y:y+h, x:x+w]
#                 if face_region.size > 0:
#                     face_resized = cv2.resize(face_region, (160, 160))
#                     face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
#                     face_tensor = face_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')

#                     # Generate embedding
#                     with torch.no_grad():
#                         embedding = siamese_net(face_tensor).squeeze()
                    
#                     # Recognize face
#                     recognized_label, similarity = recognize_face_for_image(embedding, embeddings)

#                     # Draw bounding box and label
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#                     cv2.putText(frame,f"{recognized_label} ({similarity:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
#         # Display frame
#         cv2.imshow('Real-time Face Recognition', frame)
#         frame_count += 1
        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     cap.release()
#     cv2.destroyAllWindows()




# def decode_image(image) :
#     scale_factor = 0.5
#     decoded_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

#     return decoded_image


# def save_new_user_embedding(embedding, name) :
#     print(embedding, name)

# # Serve HTML and JavaScript for webcam
# @app.route('/')
# def index():
#     return render_template('index.html')  # HTML for capturing webcam input

# # Route to capture and process image
# @app.route('/capture', methods=['POST'])
# def capture_image():
#     # Resize frame for faster processing
    
#     image_data = request.json['image']  # Get base64 encoded image from client
    

#     # Remove the data URL prefix if it's present
#     if image_data.startswith("data:image"):
#         image_data = image_data.split(",")[1]  # Remove the prefix part
    
#     # Decode the base64 image data
#     try:
#         image_bytes = base64.b64decode(image_data)
#     except base64.binascii.Error:
#         return "Invalid base64 string", 400
    
#     # Try to open the image from the bytes
#     try:
#         image = Image.open(io.BytesIO(image_bytes))
#         image = image.convert('RGB')  # Ensure image is in RGB format

#         # Convert the PIL image to a numpy array
#         image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#         # image.show()  # Open the image in the default viewer if running locally
        
#         # Optionally, save the image to verify it
#         image.save("uploaded_image.png")
#     except Image.UnidentifiedImageError:
#         return "Cannot identify image file. Make sure the data is a valid image.", 400

    
#     gray = decode_image(image_np)  # Decode and convert to OpenCV format

#     #Check if user is recognized
#     embedding = extract_face_embedding(gray, image_data)
#     recognized_label, similarity = recognize_face_for_image(embedding, embeddings_dict)
#     # print(len(embeddings_dict))
#     # print(similarity)
#     # print(recognized_label)

#     if similarity > 0.6:  # If similarity is high enough, mark as present
#         return jsonify({'status': 'recognized', 'name': recognized_label, 'similarity_score' : similarity})
#     else:
#         # Save new user data if unrecognized
#         save_new_user_embedding(embedding, "new_user_name")  # Implement function to save new user
#         return jsonify({'status': 'new_user', 'name': 'New user', 'similarity_score' : similarity})

# if __name__ == "__main__":
#     app.run(debug=True)


@app.route('/system')
def index():
    return render_template('index.html')


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/addstudent')
def add_student():
    return render_template('add_student.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/check_face', methods=['POST'])
def check_face():
    try:
        # Get image file from request
        image_file = request.files['image']
        
        # Convert to OpenCV format
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        return jsonify({
            'face_detected': len(faces) > 0
        })
        
    except Exception as e:
        print(f"Error checking face: {str(e)}")
        return jsonify({
            'face_detected': False
        })
    
@app.route("/capture-and-add", methods = ['POST'])
def capture_and_add():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Convert base64 to image
        image_data = image_data.replace('data:image/jpeg;base64,', '')
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Face detection and recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({
                'status': 'no_face_detected',
                'name': 'No face detected',
                'similarity_score': 0.0
            })
        
        # Process the first detected face
        x, y, w, h = faces[0]
        face_region = frame[y:y+h, x:x+w]
        
        # Preprocess face
        face_resized = cv2.resize(face_region, (160, 160))
        face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
        face_tensor = face_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate embedding
        with torch.no_grad():
            embedding = siamese_net(face_tensor).squeeze()

        
    except : 
        print("")


@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Convert base64 to image
        image_data = image_data.replace('data:image/jpeg;base64,', '')
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Face detection and recognition
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({
                'status': 'no_face_detected',
                'name': 'No face detected',
                'similarity_score': 0.0
            })
        
        # Process the first detected face
        x, y, w, h = faces[0]
        face_region = frame[y:y+h, x:x+w]
        
        # Preprocess face
        face_resized = cv2.resize(face_region, (160, 160))
        face_tensor = transforms.ToTensor()(face_resized).unsqueeze(0).to(torch.float32)
        face_tensor = face_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate embedding
        with torch.no_grad():
            embedding = siamese_net(face_tensor).squeeze()
        
        # Recognize face
        recognized_label, similarity = recognize_face_for_image(embedding, embeddings_dict)
        
        if similarity > 0.6:
            return jsonify({
                'status': 'recognized',
                'name': recognized_label,
                'similarity_score': float(similarity)
            })
        else:
            return jsonify({
                'status': 'new_user',
                'name': 'Unknown',
                'similarity_score': float(similarity)
            })
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Load your face detection cascade and Siamese network here
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Load your siamese_net and embeddings_dict here
    
    app.run(debug=True)