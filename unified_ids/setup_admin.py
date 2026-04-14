"""
Admin Setup Script — run ONCE to register your face + password.
Usage:  python setup_admin.py

Requirements:
    pip install face_recognition bcrypt opencv-python
"""

import cv2
import face_recognition
import bcrypt
import json
import os
import getpass
import numpy as np

USERS_FILE = 'users.json'

def capture_face():
    print("\n📷  Position your face in the webcam window.")
    print("   Press SPACE to capture, ESC to cancel.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check it is connected and not in use.")

    encoding = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw guide box
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        box_size = 220
        cv2.rectangle(frame,
                      (cx - box_size, cy - box_size),
                      (cx + box_size, cy + box_size),
                      (0, 220, 255), 2)
        cv2.putText(frame, "Center your face  |  SPACE=capture  ESC=cancel",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        cv2.imshow("Admin Face Registration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC
            print("Cancelled.")
            break

        if key == 32:   # SPACE
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb)

            if not locations:
                print("  ✗ No face detected — try again.")
                continue

            encodings = face_recognition.face_encodings(rgb, locations)
            if encodings:
                encoding = encodings[0].tolist()
                print("  ✓ Face captured successfully!")
                break

    cap.release()
    cv2.destroyAllWindows()
    return encoding


def setup():
    print("=" * 50)
    print("  IDS UNIFIED — ADMIN SETUP")
    print("=" * 50)

    username = input("\nEnter admin username: ").strip()
    if not username:
        print("Username cannot be empty.")
        return

    password = getpass.getpass("Enter admin password: ")
    confirm  = getpass.getpass("Confirm password   : ")

    if password != confirm:
        print("Passwords do not match.")
        return

    if len(password) < 6:
        print("Password must be at least 6 characters.")
        return

    # Hash password
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    # Capture face
    print("\nNow we'll capture your face encoding...")
    face_enc = capture_face()

    if face_enc is None:
        print("Face registration failed. Aborting.")
        return

    # Save to users.json
    users = {}
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)

    users[username] = {
        'password_hash': hashed,
        'face_encoding': face_enc,
    }

    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

    print(f"\n✓ Admin '{username}' registered successfully!")
    print(f"✓ Saved to {USERS_FILE}")
    print("\nYou can now run: python app.py")


if __name__ == '__main__':
    setup()
