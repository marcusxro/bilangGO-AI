import cv2
import numpy as np
import json
import os
from datetime import datetime

# Configuration
VECTORS_FILE = 'criminal_vectors.json'
FACE_DETECTION_SCALE = 0.5  # Smaller=faster but less accurate
MATCH_THRESHOLD = 0.5  # Lower=more strict matching

class FaceVerificationSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_face_vectors()
        
    def load_face_vectors(self):
        """Load face vectors from JSON file"""
        if os.path.exists(VECTORS_FILE):
            with open(VECTORS_FILE, 'r') as f:
                data = json.load(f)
                vectors = data.get('vectors', {})
                for criminal_id, criminal_data in vectors.items():
                    self.known_face_encodings.append(np.array(criminal_data['vector']))
                    self.known_face_names.append(criminal_data['name'])
                    self.known_face_ids.append(int(criminal_id))
            print(f"Loaded {len(self.known_face_encodings)} face vectors")
        else:
            raise FileNotFoundError(f"Face vectors file {VECTORS_FILE} not found")

    def run_verification(self):
        """Run real-time face verification"""
        import face_recognition  
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                break

          
            small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
            
  
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  
            

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            scaled_face_locations = []
            for (top, right, bottom, left) in face_locations:
                scaled_top = int(top / FACE_DETECTION_SCALE)
                scaled_right = int(right / FACE_DETECTION_SCALE)
                scaled_bottom = int(bottom / FACE_DETECTION_SCALE)
                scaled_left = int(left / FACE_DETECTION_SCALE)
                scaled_face_locations.append((scaled_top, scaled_right, scaled_bottom, scaled_left))

            for (top, right, bottom, left), face_encoding in zip(scaled_face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=MATCH_THRESHOLD)
                
                name = "Unknown"
                criminal_id = "N/A"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    criminal_id = self.known_face_ids[first_match_index]
                
        
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
              
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{name} (ID: {criminal_id})", (left + 6, bottom - 6), 
                            font, 0.5, (255, 255, 255), 1)

            cv2.imshow('Face Verification', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        import face_recognition
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "face_recognition"])
        import face_recognition
    
    verifier = FaceVerificationSystem()
    verifier.run_verification()