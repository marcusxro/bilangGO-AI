from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import face_recognition
import numpy as np
import mysql.connector
import os
import logging
from pydantic import BaseModel
from typing import Optional
import json
import io
import datetime



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  
    'database': 'bilanggo' 
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VerificationResult(BaseModel):
    match: bool
    criminal_id: Optional[int]
    confidence: float


def get_db_connection():
    try:
        connection = mysql.connector.connect(**MYSQL_CONFIG)
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def save_face_data(criminal_id: int, face_vector: np.ndarray):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        

        face_vector_list = face_vector.tolist()
        
        query = """
        UPDATE criminals 
        SET face_vector = %s
        WHERE id = %s
        """
        cursor.execute(query, (json.dumps(face_vector_list), criminal_id))
        connection.commit()
        
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        logger.error(f"Error saving face data: {e}")
        return False

def get_criminal_by_id(criminal_id: int):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = "SELECT * FROM criminals WHERE id = %s"
        cursor.execute(query, (criminal_id,))
        result = cursor.fetchone()
        
        if result and result.get('face_vector'):
            if isinstance(result['face_vector'], str):
                result['face_vector'] = json.loads(result['face_vector'])
        
        cursor.close()
        connection.close()
        return result
    except Exception as e:
        logger.error(f"Error fetching criminal data: {e}")
        return None

def get_all_criminals_with_faces():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = "SELECT id, name, face_vector FROM criminals WHERE face_vector IS NOT NULL"
        cursor.execute(query)
        results = cursor.fetchall()
        

        for criminal in results:
            if isinstance(criminal['face_vector'], str):
                criminal['face_vector'] = json.loads(criminal['face_vector'])
        
        cursor.close()
        connection.close()
        return results
    except Exception as e:
        logger.error(f"Error fetching criminals with face data: {e}")
        return []


@app.post("/register-face/")
async def register_face(
    file: UploadFile = File(...), 
    criminal_id: int = Form(...)
):
  
    os.makedirs("uploads", exist_ok=True)
    img_path = f"uploads/{file.filename}"
    
 
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
 
        criminal = get_criminal_by_id(criminal_id)
        if not criminal:
            raise HTTPException(status_code=404, detail="Criminal not found")

  
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
 
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

      
        if not save_face_data(criminal_id, face_encoding):
            raise HTTPException(status_code=500, detail="Failed to store face data")

        return {
            "criminal_id": criminal_id,
            "name": criminal['name'],
            "message": "Face data registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        
        if os.path.exists(img_path):
            os.remove(img_path)

@app.post("/verify-face/", response_model=VerificationResult)
async def verify_face(file: UploadFile = File(...)):
  
    os.makedirs("uploads", exist_ok=True)
    img_path = f"uploads/{file.filename}"
    
 
    with open(img_path, "wb") as f:
        f.write(await file.read())

    try:
        
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            return VerificationResult(match=False, confidence=0.0)
            
     
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]

       
        criminals = get_all_criminals_with_faces()
        if not criminals:
            return VerificationResult(match=False, confidence=0.0)

     
        best_match = None
        best_confidence = 0.0

        for criminal in criminals:
            try:
                stored_encoding = np.array(criminal['face_vector'])
                face_distances = face_recognition.face_distance([stored_encoding], face_encoding)
                confidence = 1 - face_distances[0]

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = criminal['id']
            except Exception as e:
                logger.error(f"Error comparing faces for criminal {criminal['id']}: {e}")
                continue

        
        match = best_confidence >= 0.6
        return VerificationResult(
            match=match,
            criminal_id=best_match if match else None,
            confidence=float(best_confidence))
            
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
       
        if os.path.exists(img_path):
            os.remove(img_path)
@app.post("/generate-face-vector")
async def generate_face_vector(file: UploadFile = File(...)):
    try:
 
        criminals = get_all_criminals_with_faces()
       
        face_vectors_data = {
            "vectors": {},
            "last_update": datetime.datetime.now().isoformat()
        }
        
        for criminal in criminals:
            if criminal.get('face_vector'):
                face_vectors_data["vectors"][str(criminal['id'])] = {
                    "vector": criminal['face_vector'],
                    "name": criminal.get('name', 'Unknown'),
                    "updated_at": criminal.get('updated_at', datetime.datetime.now().isoformat())
                }
  
        with open('face_vectors.json', 'w') as f:
            json.dump(face_vectors_data, f, indent=4)
        
        logger.info("Successfully updated face_vectors.json with latest data from MySQL")
        
       
        contents = await file.read()
        image = face_recognition.load_image_file(io.BytesIO(contents))
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        return {
            "message": "Face vectors database updated and new vector generated",
            "vector": face_encoding.tolist(),
            "database_updated": True
        }
        
    except Exception as e:
        logger.error(f"Error in generate-face-vector: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
