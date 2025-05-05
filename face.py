import cv2
import numpy as np
import json
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import face_recognition
from io import BytesIO
from PIL import Image

VECTORS_FILE = 'face_vectors.json'
FACE_DETECTION_SCALE = 0.5
MATCH_THRESHOLD = 0.5

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FaceResult(BaseModel):
    top: int
    right: int
    bottom: int
    left: int
    name: str
    criminal_id: int | None


class FaceVerificationSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_face_vectors()

    def load_face_vectors(self):
        if os.path.exists(VECTORS_FILE):
            with open(VECTORS_FILE, 'r') as f:
                data = json.load(f)
                vectors = data.get('vectors', {})
                for criminal_id, criminal_data in vectors.items():
                    self.known_face_encodings.append(np.array(criminal_data['vector']))
                    self.known_face_names.append(criminal_data['name'])
                    self.known_face_ids.append(int(criminal_id))
        else:
            raise FileNotFoundError(f"Face vectors file {VECTORS_FILE} not found")

    def process_image(self, image_np) -> List[FaceResult]:
  
        small_frame = cv2.resize(image_np, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
          
            top = int(top / FACE_DETECTION_SCALE)
            right = int(right / FACE_DETECTION_SCALE)
            bottom = int(bottom / FACE_DETECTION_SCALE)
            left = int(left / FACE_DETECTION_SCALE)

            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=MATCH_THRESHOLD)
            name = "Unknown"
            criminal_id = None
            if True in matches:
                index = matches.index(True)
                name = self.known_face_names[index]
                criminal_id = self.known_face_ids[index]

            results.append(FaceResult(
                top=top, right=right, bottom=bottom, left=left,
                name=name, criminal_id=criminal_id
            ))
        return results


verifier = FaceVerificationSystem()


@app.post("/verify_faces", response_model=List[FaceResult])
async def verify_faces(file: UploadFile = File(...)):
    
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

   
    results = verifier.process_image(image_np)
    return results
