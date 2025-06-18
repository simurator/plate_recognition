import cv2
import numpy as np
import imutils
import easyocr
import requests
import tkinter as tk
from PIL import Image, ImageTk
import threading
import winsound  # Tylko Windows
import re
import json


# === Funkcja Levenshteina ===
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def is_similar(plate1, plate2, max_distance=2):
    return levenshtein_distance(plate1, plate2) <= max_distance

def normalize_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# === OCR Reader ===
reader = easyocr.Reader(['en'])

# === Pobierz tablice z API ===
recognized_plates = set()

def fetch_plates_from_api():
    try:
        response = requests.get("https://api-site-blush.vercel.app/api/data")
        if response.status_code == 200:
            data = json.loads(response.text)
            message = data.get("message", "")
            words = message.split()
            for word in words:
                normalized = normalize_text(word)
                if normalized:
                    recognized_plates.add(normalized)
            print(f"✅ Załadowano {len(recognized_plates)} tablic z API.")
        else:
            print(f"❌ Błąd API: {response.status_code}")
    except Exception as e:
        print(f"❌ Błąd podczas pobierania danych z API: {e}")



# === GUI ===
root = tk.Tk()
root.title("Rozpoznawanie Tablic - Kamera")
label = tk.Label(root)
label.pack()

def periodic_fetch():
    recognized_plates.clear()
    fetch_plates_from_api()
    root.after(300_000, periodic_fetch)  # co 5 minut

periodic_fetch()

# === Kamera ===
cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    display_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        result = reader.readtext(cropped_image)
        for detection in result:
            raw_text = detection[1]
            confidence = detection[2]
            text = normalize_text(raw_text)

            if confidence > 0.5 and text:
                is_duplicate = any(is_similar(text, existing) for existing in recognized_plates)
                if is_duplicate:
                    print(f"🚨 Wykryto istniejacą tablicę: {text}")
                    threading.Thread(target=winsound.Beep, args=(1000, 300), daemon=True).start()

                cv2.putText(display_frame, text, (location[0][0][0], location[1][0][1] + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(display_frame, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

    img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    root.after(30, update_frame)

update_frame()
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.mainloop()
