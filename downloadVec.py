import mysql.connector
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL Configuration
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',  # Replace with your MySQL password
    'database': 'bilanggo'
}

def get_db_connection():
    try:
        return mysql.connector.connect(**MYSQL_CONFIG)
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def get_criminals_with_vectors():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        query = """
        SELECT id, name, face_vector 
        FROM criminals 
        WHERE face_vector IS NOT NULL 
        AND face_vector != 'NULL'
        AND JSON_LENGTH(face_vector) > 0
        """
        cursor.execute(query)
        results = cursor.fetchall()
        
        output = {}
        for row in results:
            try:
                if isinstance(row['face_vector'], str):
                    vector = json.loads(row['face_vector'])
                else:
                    vector = row['face_vector']
                
                if vector: 
                    output[str(row['id'])] = {
                        "vector": vector,
                        "name": row['name']
                    }
            except json.JSONDecodeError:
                logger.warning(f"Invalid face vector for ID {row['id']}")
                continue
        
        cursor.close()
        conn.close()
        return output
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        return {}

def save_vectors_to_file():
    vectors = get_criminals_with_vectors()
    
    if not vectors:
        logger.error("No valid face vectors found in database!")
        return False
    
    try:
        with open('criminal_vectors.json', 'w') as f:
            json.dump({"vectors": vectors}, f, indent=4)
        
        logger.info(f"Saved {len(vectors)} criminal face vectors")
        return True
    except Exception as e:
        logger.error(f"File save error: {e}")
        return False

if __name__ == "__main__":
    if save_vectors_to_file():
        print("Successfully exported face vectors to criminal_vectors.json")
    else:
        print("Failed to export face vectors. Check logs for details.")