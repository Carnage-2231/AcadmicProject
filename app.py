from flask import Flask, Response, jsonify, request #rai
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import threading
import time
from datetime import datetime

import torch.nn.functional as F

import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt



app = Flask(__name__)
CORS(app)

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset"
MODEL_PATH = "tgl_net.pth"
FRAMES = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global state
model = None
labels = []
camera = None
is_camera_on = True
prediction_sequence = []
current_prediction = ""
training_in_progress = False
training_log = []
auto_trained = False

# ==============================
# MODEL DEFINITION
# ==============================
# class TGLNet(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.temporal_fc = nn.Linear(126, 128)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=128,
#                 nhead=8,
#                 batch_first=True
#             ),
#             num_layers=2
#         )
#         self.classifier = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.temporal_fc(x)
#         x = self.transformer(x)
#         x = x.mean(dim=1)
#         return self.classifier(x)

# ==============================
# REAL HAND ADJACENCY (42 nodes for 2 hands)
# ==============================
def get_adjacency():
    A = np.zeros((42, 42))

    # 21 landmarks per hand
    edges = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]

    # First hand
    for i,j in edges:
        A[i][j] = 1
        A[j][i] = 1

    # Second hand (shift by 21)
    for i,j in edges:
        A[i+21][j+21] = 1
        A[j+21][i+21] = 1

    return torch.tensor(A, dtype=torch.float32)

# ==============================
# GRAPH CONVOLUTION LAYER
# ==============================
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.A = get_adjacency().to(DEVICE)

    def forward(self, x):
        # x: (B, T, J, C)
        B, T, J, C = x.shape
        x = torch.einsum('btjc,jk->btkc', x, self.A)
        x = self.fc(x)
        return x



# ==============================
# ST-GCN BLOCK
# ==============================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels)
        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, T, J, C)
        x = self.gcn(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, T, J)
        x = self.temporal_conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)  # back to (B, T, J, C)
        return x


# ==============================
# GRAPH TRANSFORMER MODEL
# ==============================
class GraphTransformerNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stgcn1 = STGCNBlock(3, 64)
        self.stgcn2 = STGCNBlock(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, J, C)

        x = self.stgcn1(x)
        x = self.stgcn2(x)

        # Global spatial pooling
        x = x.mean(dim=2)  # (B, T, C)

        x = self.transformer(x)

        x = x.mean(dim=1)  # temporal pooling

        return self.classifier(x)

# ==============================
# CAMERA MANAGEMENT
# ==============================
def get_camera():
    global camera, is_camera_on
    if camera is None or not is_camera_on:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            # Try with DSHOW on Windows
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = None
            print("❌ Camera not accessible")
            return None
        is_camera_on = True
        print("📹 Camera started")
    return camera

def release_camera():
    global camera, is_camera_on
    if camera:
        camera.release()
        camera = None
        is_camera_on = False
        print("📹 Camera stopped")

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ==============================
# AUTO-INITIALIZATION
# ==============================
def check_and_initialize():
    """Check system state and auto-initialize missing components"""
    global labels, auto_trained
    
    print("\n" + "="*60)
    print("🚀 AUTOMATIC SYSTEM INITIALIZATION")
    print("="*60)
    
    # Step 1: Check dataset directory
    if not os.path.exists(DATASET_DIR):
        print(f"📁 Creating dataset directory: {DATASET_DIR}")
        os.makedirs(DATASET_DIR, exist_ok=True)
    else:
        print(f"✅ Dataset directory exists: {DATASET_DIR}")
    
    # Step 2: Check for existing gestures
    labels = sorted([d for d in os.listdir(DATASET_DIR) 
                    if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    if len(labels) == 0:
        print("📝 No gestures found in dataset")
        print("   → Use the Dataset Creator to record gestures")
        print("   → Or add gesture folders manually to 'dataset/'")
        return False
    else:
        print(f"✅ Found {len(labels)} gesture(s): {labels}")
        
        # Check sample counts
        total_samples = 0
        for label in labels:
            folder = os.path.join(DATASET_DIR, label)
            samples = len([f for f in os.listdir(folder) if f.endswith('.npy')])
            total_samples += samples
            print(f"   - {label}: {samples} samples")
        
        if total_samples == 0:
            print("⚠️  Gesture folders exist but contain no samples")
            return False
    
    # Step 3: Check for trained model
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model file not found: {MODEL_PATH}")
        print("🤖 AUTO-TRAINING INITIATED...")
        
        success = auto_train_model()
        if success:
            auto_trained = True
            print("✅ Auto-training completed successfully!")
            return load_model()
        else:
            print("❌ Auto-training failed")
            return False
    else:
        print(f"✅ Model file exists: {MODEL_PATH}")
        return load_model()

def load_model():
    """Load the trained model"""
    global model, labels
    
    try:
        # Get labels from dataset
        labels = sorted([d for d in os.listdir(DATASET_DIR) 
                        if os.path.isdir(os.path.join(DATASET_DIR, d))])
        
        if len(labels) == 0:
            print("⚠️  No gesture labels found")
            return False
        
        if not os.path.exists(MODEL_PATH):
            print(f"⚠️  Model file not found: {MODEL_PATH}")
            return False
        
        # Load model
        # model = TGLNet(len(labels)).to(DEVICE)
        model = GraphTransformerNet(len(labels)).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   - Device: {DEVICE}")
        print(f"   - Classes: {len(labels)}")
        print(f"   - Labels: {labels}")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        return False


from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns


def auto_train_model():
    """
    Advanced Training Pipeline:
    - Group K-Fold Cross Validation (Video Safe)
    - F1 Score Reporting
    - Confusion Matrix Saving
    - TensorBoard Logging
    - Best Model Saving
    """

    global model, labels, training_log
    training_log = []

    try:
        # -----------------------------
        # Load Dataset
        # -----------------------------
        labels = sorted([
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ])

        if len(labels) < 2:
            training_log.append("❌ Need at least 2 gesture classes.")
            return False

        label_map = {label: i for i, label in enumerate(labels)}

        X, y, groups = [], [], []

        training_log.append("📂 Loading dataset...")

        video_counter = 0

        for label in labels:
            folder = os.path.join(DATASET_DIR, label)
            files = [f for f in os.listdir(folder) if f.endswith(".npy")]

            for file in files:
                data = np.load(os.path.join(folder, file))
                X.append(data)
                y.append(label_map[label])

                # Each file = one video group
                groups.append(video_counter)
                video_counter += 1

        X = np.array(X).reshape(-1, FRAMES, 42, 3)
        y = np.array(y)
        groups = np.array(groups)

        if len(X) < 10:
            training_log.append("❌ Not enough samples.")
            return False

        training_log.append(f"📊 Total videos: {len(X)}")

        # -----------------------------
        # Group K-Fold Setup
        # -----------------------------
        gkf = GroupKFold(n_splits=5)
        fold_f1_scores = []
        best_f1 = 0

        for fold, (train_ids, val_ids) in enumerate(gkf.split(X, y, groups)):

            training_log.append(f"\n🔁 Fold {fold+1}/5")

            model = GraphTransformerNet(len(labels)).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            writer = SummaryWriter(log_dir=f"runs/group_fold_{fold+1}")

            X_train, X_val = X[train_ids], X[val_ids]
            y_train, y_val = y[train_ids], y[val_ids]

            train_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            )

            val_dataset = torch.utils.data.TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            )

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)

            # -----------------------------
            # Training Loop
            # -----------------------------
            epochs = 20

            for epoch in range(epochs):

                model.train()
                total_loss = 0

                for xb, yb in train_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                    optimizer.zero_grad()
                    output = model(xb)
                    loss = criterion(output, yb)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                writer.add_scalar("Loss/train", avg_loss, epoch)

                training_log.append(
                    f"Fold {fold+1} | Epoch {epoch+1} | Loss: {avg_loss:.4f}"
                )

            # -----------------------------
            # Validation
            # -----------------------------
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(DEVICE)
                    output = model(xb)

                    probs = torch.softmax(output, dim=1)
                    preds = torch.argmax(probs, dim=1)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(yb.numpy())

            f1 = f1_score(all_labels, all_preds, average="weighted")
            fold_f1_scores.append(f1)

            writer.add_scalar("F1/validation", f1, 0)
            training_log.append(f"✅ Fold {fold+1} F1 Score: {f1:.4f}")

            # Save best model across folds
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), MODEL_PATH)
                training_log.append("💾 Best model updated.")

            # -----------------------------
            # Confusion Matrix
            # -----------------------------
            cm = confusion_matrix(all_labels, all_preds)

            plt.figure(figsize=(6,5))
            sns.heatmap(cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=labels,
                        yticklabels=labels)

            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - Fold {fold+1}")

            cm_path = f"confusion_matrix_group_fold_{fold+1}.png"
            plt.savefig(cm_path)
            plt.close()

            training_log.append(f"📊 Confusion matrix saved: {cm_path}")
            writer.close()

        # -----------------------------
        # Final Average F1
        # -----------------------------
        avg_f1 = np.mean(fold_f1_scores)

        training_log.append(f"\n🎯 Average F1 Score (Group 5-Fold): {avg_f1:.4f}")
        print(f"\n🎯 Average F1 Score: {avg_f1:.4f}")

        return True

    except Exception as e:
        training_log.append(f"❌ Training Error: {str(e)}")
        return False

# ==============================
# API ENDPOINTS
# ==============================

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """Get complete system status"""
    dataset_exists = os.path.exists(DATASET_DIR)
    model_exists = os.path.exists(MODEL_PATH)
    
    gesture_count = 0
    total_samples = 0
    gesture_details = []
    
    if dataset_exists:
        gestures = [d for d in os.listdir(DATASET_DIR) 
                   if os.path.isdir(os.path.join(DATASET_DIR, d))]
        gesture_count = len(gestures)
        
        for gesture in gestures:
            gesture_path = os.path.join(DATASET_DIR, gesture)
            samples = len([f for f in os.listdir(gesture_path) if f.endswith('.npy')])
            total_samples += samples
            gesture_details.append({
                "name": gesture,
                "samples": samples
            })
    
    status = {
        "dataset_folder_exists": dataset_exists,
        "model_file_exists": model_exists,
        "model_loaded": model is not None,
        "gesture_count": gesture_count,
        "total_samples": total_samples,
        "gestures": gesture_details,
        "labels": labels,
        "camera_on": is_camera_on,
        "ready_for_prediction": model is not None and len(labels) > 0,
        "training_in_progress": training_in_progress,
        "auto_trained": auto_trained,
        "device": str(DEVICE),
        "setup_steps": []
    }
    
    # Provide setup steps
    if not dataset_exists or gesture_count == 0:
        status["setup_steps"].append({
            "step": 1,
            "action": "Create datasets",
            "description": "Use the Dataset Creator tab to record gesture samples",
            "completed": False
        })
    else:
        status["setup_steps"].append({
            "step": 1,
            "action": "Create datasets",
            "description": f"Found {gesture_count} gesture(s) with {total_samples} samples",
            "completed": True
        })
    
    if not model_exists:
        status["setup_steps"].append({
            "step": 2,
            "action": "Train model",
            "description": "Model will auto-train when you create datasets, or click 'Train Now'",
            "completed": False
        })
    else:
        status["setup_steps"].append({
            "step": 2,
            "action": "Train model",
            "description": "Model file exists" + (" (auto-trained)" if auto_trained else ""),
            "completed": True
        })
    
    if model is None and model_exists and len(labels) > 0:
        status["setup_steps"].append({
            "step": 3,
            "action": "Reload model",
            "description": "Click 'Reload Model' button to load the trained model",
            "completed": False
        })
    elif model is not None:
        status["setup_steps"].append({
            "step": 3,
            "action": "Ready for prediction",
            "description": f"Model loaded with {len(labels)} gesture classes",
            "completed": True
        })
    
    return jsonify(status)

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    cam = get_camera()
    if cam is None:
        return jsonify({"status": "error", "message": "Camera not accessible"}), 500
    return jsonify({"status": "success", "message": "Camera started"})

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    release_camera()
    return jsonify({"status": "success", "message": "Camera stopped"})

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    return jsonify({"is_on": is_camera_on})

@app.route('/api/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """Generate video frames with prediction overlay (stable version)"""
    global prediction_sequence, current_prediction

    cam = get_camera()
    if cam is None:
        return

    mp_hands_local = mp.solutions.hands

    # Initialize MediaPipe INSIDE generator (fixes timestamp issue)
    with mp_hands_local.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while is_camera_on:
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
            except Exception as e:
                print("MediaPipe error:", e)
                continue

            # Draw hand landmarks
            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS
                    )

                # Collect landmarks for prediction
                if model is not None:
                    landmarks = []

                    for hand_lms in result.multi_hand_landmarks:
                        for lm in hand_lms.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])

                    # Pad to 2 hands (42 landmarks × 3 values = 126)
                    while len(landmarks) < 126:
                        landmarks.extend([0.0, 0.0, 0.0])

                    prediction_sequence.append(landmarks)

                    if len(prediction_sequence) == FRAMES:
                        try:
                            input_array = np.array(prediction_sequence).reshape(1, FRAMES, 42, 3)
                            input_tensor = torch.tensor(input_array, dtype=torch.float32).to(DEVICE)

                            with torch.no_grad():
                                output = model(input_tensor)
                                probs = torch.softmax(output, dim=1)
                                confidence, pred_index = torch.max(probs, 1)

                                confidence_value = confidence.item()
                                pred_index = pred_index.item()

                                if confidence_value > 0.70:   # confidence threshold
                                    current_prediction = f"{labels[pred_index]} ({confidence_value*100:.1f}%)"
                                else:
                                    current_prediction = "Uncertain"


                        except Exception as e:
                            print("Model error:", e)

                        prediction_sequence = []

            # Add prediction overlay
            if current_prediction and model is not None:
                cv2.rectangle(frame, (10, 10), (500, 90), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    f"Prediction: {current_prediction}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )

            # Encode frame safely
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

@app.route('/api/prediction/current', methods=['GET'])
def get_current_prediction():
    return jsonify({
        "prediction": current_prediction,
        "frames_collected": len(prediction_sequence),
        "frames_required": FRAMES
    })

@app.route('/api/dataset/gestures', methods=['GET'])
def get_gestures():
    gestures = []
    if os.path.exists(DATASET_DIR):
        gestures = sorted([d for d in os.listdir(DATASET_DIR) 
                          if os.path.isdir(os.path.join(DATASET_DIR, d))])
    return jsonify({"gestures": gestures})

@app.route('/api/dataset/create', methods=['POST'])
def create_dataset():
    data = request.json
    gesture_name = data.get('gesture_name')
    num_samples = data.get('num_samples', 20)
    frames_per_sample = data.get('frames_per_sample', 30)
    
    if not gesture_name:
        return jsonify({"status": "error", "message": "Gesture name required"}), 400
    
    gesture_path = os.path.join(DATASET_DIR, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)
    
    existing_samples = len([f for f in os.listdir(gesture_path) if f.endswith('.npy')])
    
    return jsonify({
        "status": "success",
        "gesture_path": gesture_path,
        "existing_samples": existing_samples,
        "message": f"Ready to record {num_samples} samples"
    })

@app.route('/api/dataset/record_sample', methods=['POST'])
def record_sample():
    data = request.json
    gesture_name = data.get('gesture_name')
    sample_index = data.get('sample_index', 0)
    frames_required = data.get('frames_required', 30)
    
    if not gesture_name:
        return jsonify({"status": "error", "message": "Gesture name required"}), 400
    
    gesture_path = os.path.join(DATASET_DIR, gesture_name)
    os.makedirs(gesture_path, exist_ok=True)
    
    cam = get_camera()
    if cam is None:
        return jsonify({"status": "error", "message": "Camera not available"}), 500
    
    sequence = []
    frames_collected = 0
    max_attempts = frames_required * 3
    attempts = 0
    
    while frames_collected < frames_required and attempts < max_attempts:
        ret, frame = cam.read()
        attempts += 1
        
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        if result.multi_hand_landmarks:
            landmarks = []
            for hand_lms in result.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            
            while len(landmarks) < 126:
                landmarks.extend([0.0, 0.0, 0.0])
            
            sequence.append(landmarks)
            frames_collected += 1
    
    if frames_collected == frames_required:
        sample_path = os.path.join(gesture_path, f"sample_{sample_index}.npy")
        np.save(sample_path, np.array(sequence))
        
        return jsonify({
            "status": "success",
            "message": f"Sample {sample_index} recorded",
            "frames_collected": frames_collected
        })
    else:
        return jsonify({
            "status": "error",
            "message": f"Failed to collect enough frames. Got {frames_collected}/{frames_required}"
        }), 400

@app.route('/api/dataset/delete', methods=['POST'])
def delete_gesture():
    data = request.json
    gesture_name = data.get('gesture_name')
    
    if not gesture_name:
        return jsonify({"status": "error", "message": "Gesture name required"}), 400
    
    gesture_path = os.path.join(DATASET_DIR, gesture_name)
    
    if os.path.exists(gesture_path):
        import shutil
        shutil.rmtree(gesture_path)
        return jsonify({"status": "success", "message": f"Deleted {gesture_name}"})
    else:
        return jsonify({"status": "error", "message": "Gesture not found"}), 404

@app.route('/api/model/train', methods=['POST'])
def train_model_endpoint():
    global training_in_progress
    
    if training_in_progress:
        return jsonify({
            "status": "error",
            "message": "Training already in progress"
        }), 400
    
    # Start training in background thread
    def train_background():
        global training_in_progress
        training_in_progress = True
        try:
            success = auto_train_model()
            if success:
                load_model()
        finally:
            training_in_progress = False
    
    thread = threading.Thread(target=train_background)
    thread.start()
    
    return jsonify({
        "status": "success",
        "message": "Training started in background. Check /api/training/status for progress."
    })

@app.route('/api/model/confusion', methods=['GET'])
def get_confusion_matrix():
    images = [f for f in os.listdir('.') if f.startswith("confusion_fold")]
    return jsonify({
        "confusion_matrices": images
    })


@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    return jsonify({
        "training_in_progress": training_in_progress,
        "training_log": training_log
    })

@app.route('/api/model/reload', methods=['POST'])
def reload_model():
    success = load_model()
    if success:
        return jsonify({
            "status": "success",
            "message": "Model reloaded successfully",
            "labels": labels
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Failed to reload model"
        }), 500

# ==============================
# STARTUP
# ==============================
if __name__ == '__main__':
    print("\n" + "🤖 "*30)
    print("   AUTOMATIC SIGN LANGUAGE RECOGNITION SERVER")
    print("🤖 "*30 + "\n")
    
    # Run auto-initialization
    check_and_initialize()
    
    print("🌐 Starting Flask server...")
    print("   Access at: http://localhost:5000")
    print("   Frontend: http://localhost:5173 (if running)")
    print("\n" + "="*60 + "\n")
    
    # Start server
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)