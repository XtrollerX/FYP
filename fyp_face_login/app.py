import base64
import json
import os

import cv2
import face_recognition
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid


app = Flask(__name__)
CORS(app)  # allow requests from your website (localhost different port)

DB_PATH = "known_faces.json"

# ---------- ID creation ----------
def generate_user_id(db):
    """
    Generates a unique user ID using UUID.
    Example: u_f3a12b80
    """
    new_id = f"u_{uuid.uuid4().hex[:8]}"  # short, unique, no collisions
    while new_id in db:
        new_id = f"u_{uuid.uuid4().hex[:8]}"
    return new_id
# ---------- Simple "database" using JSON file ----------

def load_db():
    if not os.path.exists(DB_PATH):
        return {}
    with open(DB_PATH, "r") as f:
        raw = json.load(f)
    # convert lists back to numpy arrays
    return {uid: np.array(emb, dtype="float32") for uid, emb in raw.items()}

def save_db(db):
    serializable = {uid: emb.tolist() for uid, emb in db.items()}
    with open(DB_PATH, "w") as f:
        json.dump(serializable, f)

# ---------- Helpers ----------

def decode_base64_image(b64_str: str):
    # strip dataURL header: "data:image/jpeg;base64,..."
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def get_single_face_embedding(img_rgb):
    locations = face_recognition.face_locations(img_rgb)
    if len(locations) != 1:
        return None, len(locations)
    encoding = face_recognition.face_encodings(img_rgb, locations)[0]
    return encoding, 1

# ---------- API endpoints ----------

@app.route("/api/register_face", methods=["POST"])
def register_face():
    """
    Request JSON:
      { "image": "<base64 image>" }

    Behaviour:
      - If face already exists: returns existing user_id with is_new = False
      - If new face: generates new user_id with is_new = True
    """
    data = request.get_json(force=True)
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"success": False, "error": "image required"}), 400

    img = decode_base64_image(image_b64)
    embedding, face_count = get_single_face_embedding(img)

    if embedding is None:
        return jsonify({
            "success": False,
            "error": f"Expected 1 face, found {face_count}"
        }), 400

    db = load_db()

    # ---- 1. Check if face already exists ----
    if db:
        known_ids = list(db.keys())
        known_embeddings = np.stack([db[uid] for uid in known_ids], axis=0)

        distances = face_recognition.face_distance(known_embeddings, embedding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        best_user = known_ids[best_idx]

        SAME_FACE_THRESHOLD = 0.50

        if best_distance <= SAME_FACE_THRESHOLD:
            # Existing user
            return jsonify({
                "success": True,
                "user_id": best_user,
                "distance": best_distance,
                "is_new": False,
                "message": "Existing user recognized"
            }), 200

    # ---- 2. No similar face -> create new user ----
    new_user_id = generate_user_id(db)
    db[new_user_id] = embedding
    save_db(db)

    return jsonify({
        "success": True,
        "user_id": new_user_id,
        "is_new": True,
        "message": "New user registered"
    }), 200

@app.route("/api/verify_face", methods=["POST"])
def verify_face():
    """
    Request JSON:
      {
        "image": "<base64 dataURL from canvas.toDataURL()>"
      }

    Response (success):
      {
        "success": true,
        "user_id": "matched-user-id",
        "distance": 0.43
      }
    """
    data = request.get_json(force=True)
    image_b64 = data.get("image")

    if not image_b64:
        return jsonify({"success": False, "error": "image required"}), 400

    img = decode_base64_image(image_b64)
    embedding, face_count = get_single_face_embedding(img)

    if embedding is None:
        return jsonify({
            "success": False,
            "error": f"Expected 1 face, found {face_count}"
        }), 400

    db = load_db()
    if not db:
        return jsonify({"success": False, "error": "no registered faces"}), 400

    known_ids = list(db.keys())
    known_embeddings = np.stack([db[uid] for uid in known_ids], axis=0)

    distances = face_recognition.face_distance(known_embeddings, embedding)
    best_idx = int(np.argmin(distances))
    best_distance = float(distances[best_idx])
    best_user = known_ids[best_idx]

    THRESHOLD = 0.6  # tune this later (lower = stricter)
    if best_distance <= THRESHOLD:
        return jsonify({
            "success": True,
            "user_id": best_user,
            "distance": best_distance
        })
    else:
        return jsonify({
            "success": False,
            "error": "no matching user",
            "distance": best_distance
        }), 401

@app.route("/")
def health():
    return "Face login backend running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
