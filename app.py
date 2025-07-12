
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import mediapipe as mp
from numpy.linalg import norm

app = Flask(__name__)
CORS(app)

# Load heavy models
face_verifier = FaceAnalysis()
face_verifier.prepare(ctx_id=-1)  # CPU only
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def check_head_pose(landmarks):
    nose = landmarks.landmark[1]
    left_t = landmarks.landmark[234]
    right_t = landmarks.landmark[454]
    face_w = abs(left_t.x - right_t.x)
    nose_offset = abs(nose.x - (left_t.x + right_t.x) / 2)
    return nose_offset > face_w * 0.2

def check_eyes_closed(landmarks):
    def ear(points):
        v1 = abs(landmarks.landmark[points[1]].y - landmarks.landmark[points[5]].y)
        v2 = abs(landmarks.landmark[points[2]].y - landmarks.landmark[points[4]].y)
        h = abs(landmarks.landmark[points[0]].x - landmarks.landmark[points[3]].x)
        return (v1 + v2) / (2.0 * h)
    return (ear(LEFT_EYE) + ear(RIGHT_EYE)) / 2 < 0.2

@app.route("/analyze_frame", methods=["POST"])
def analyze_frame():
    try:
        file = request.files["frame"]
        embedding = np.frombuffer(request.files["embedding"].read(), dtype=np.float32)
        arr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = {
            "identity_mismatch": False,
            "looking_away": False,
            "multiple_faces": False,
            "eyes_closed": False
        }

        faces = face_verifier.get(frame)
        if len(faces) != 1:
            results["multiple_faces"] = True
            if len(faces) == 0:
                results["identity_mismatch"] = True
        else:
            live_emb = faces[0].embedding
            sim = np.dot(embedding, live_emb) / (norm(embedding) * norm(live_emb))
            if sim < 0.55:
                results["identity_mismatch"] = True

        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5) as fm:
            mesh_res = fm.process(rgb)
            if mesh_res.multi_face_landmarks:
                lm = mesh_res.multi_face_landmarks[0]
                results["looking_away"] = check_head_pose(lm)
                results["eyes_closed"] = check_eyes_closed(lm)

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API running"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
