import requests

# יצירת וקטורי דמה באורכים המדויקים שמסד הנתונים מצפה לקבל
face_vec = [0.1] * 512
finger_vec = [0.2] * 128

print("1. Enrolling user...")
enroll_res = requests.post("http://localhost:8000/enroll", json={
    "username": "test_user_1",
    "face_vector": face_vec,
    "fingerprint_vector": finger_vec
})
print("Enroll Response:", enroll_res.json())

print("\n2. Authenticating user...")
auth_res = requests.post("http://localhost:8000/authenticate", json={
    "face_vector": face_vec,
    "fingerprint_vector": finger_vec,
    "face_weight": 0.23,
    "finger_weight": 0.77
})
print("Auth Response:", auth_res.json())