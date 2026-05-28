import cv2
import numpy as np
import os

# ==========================================
# 1. INISIALISASI & PERSIAPAN TEMPLATE
# ==========================================
# Inisialisasi detektor SIFT.
sift = cv2.SIFT_create()

# Inisialisasi Brute-Force Matcher dengan NORM_L2 (Wajib untuk SIFT/Float Descriptor)
bf = cv2.BFMatcher(cv2.NORM_L2)

# Load gambar template referensi
img_ref = cv2.imread('image/template-uang.png')

if img_ref is None:
    print("Error: Gambar template_uang tidak ditemukan!")
    exit()

# ==========================================
# TAMBAHAN BARU: Perkecil resolusi gambar template dan buat horizontal
# ==========================================
tinggi_asli, lebar_asli = img_ref.shape[:2]

# Putar gambar menjadi horizontal jika orientasinya vertikal (tinggi > lebar)
if tinggi_asli > lebar_asli:
    img_ref = cv2.rotate(img_ref, cv2.ROTATE_90_COUNTERCLOCKWISE)
    tinggi_asli, lebar_asli = img_ref.shape[:2] # Update dimensi setelah rotasi

# Mengecilkan gambar menjadi lebar 400 piksel agar proporsional sebagai referensi
rasio = 400 / lebar_asli
dimensi_baru = (400, int(tinggi_asli * rasio))
img_ref = cv2.resize(img_ref, dimensi_baru, interpolation=cv2.INTER_AREA)
# ==========================================

# --- PREPROCESSING TEMPLATE (Materi Dosen) ---
# 1. Grayscale
gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
# 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_ref = clahe.apply(gray_ref)

# --- FILTERING TEMPLATE (SIFT) ---
gray_ref_gauss = cv2.GaussianBlur(gray_ref, (5, 5), 0)
gray_ref_bilat = cv2.bilateralFilter(gray_ref, 9, 75, 75)

# Ekstrak Keypoints (kp) dan Descriptors (des) dari Template
kp_ref_gauss, des_ref_gauss = sift.detectAndCompute(gray_ref_gauss, None)
kp_ref_bilat, des_ref_bilat = sift.detectAndCompute(gray_ref_bilat, None)

# Ambil dimensi template untuk menggambar Bounding Box nanti
h, w = gray_ref.shape
template_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

# Thumbnail template untuk ditampilkan di frame hasil
thumb_w = 310
thumb_h = int(thumb_w * (img_ref.shape[0] / img_ref.shape[1]))
template_thumb = cv2.resize(img_ref, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
thumb_scale_x = thumb_w / img_ref.shape[1]
thumb_scale_y = thumb_h / img_ref.shape[0]

def overlay_thumbnail(frame, thumb, top=10, left=10):
    h_t, w_t = thumb.shape[:2]
    frame[top:top + h_t, left:left + w_t] = thumb
    return frame

BOX_THICKNESS = 2
POINT_RADIUS = 2
FRAME_POINT_RADIUS = 2
HOLD_FRAMES = 5
MAX_MATCH_POINTS = 30
MAX_MATCH_LINES = 20
DRAW_BOX = False

def draw_match_points_on_thumb(thumb, pts, color):
    for x, y in pts[:MAX_MATCH_POINTS]:
        cx = int(x * thumb_scale_x)
        cy = int(y * thumb_scale_y)
        cv2.circle(thumb, (cx, cy), POINT_RADIUS, color, -1)
    return thumb

def draw_match_lines_from_thumb(frame, pts_ref, pts_frame, thumb_origin, color):
    ox, oy = thumb_origin
    for (rx, ry), (fx, fy) in zip(pts_ref[:MAX_MATCH_LINES], pts_frame[:MAX_MATCH_LINES]):
        sx = int(rx * thumb_scale_x) + ox
        sy = int(ry * thumb_scale_y) + oy
        cv2.line(frame, (sx, sy), (int(fx), int(fy)), color, 1, cv2.LINE_AA)

def track_sift(frame, filter_name, ref_img, ref_kp, ref_des, filter_fn, label_color, state):
    frame_out = frame.copy()

    # --- PREPROCESSING FRAME KAMERA ---
    gray_frame = cv2.cvtColor(frame_out, cv2.COLOR_BGR2GRAY)
    gray_frame = clahe.apply(gray_frame)
    gray_frame = filter_fn(gray_frame)

    # Ekstrak fitur dari frame kamera saat ini
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

    matched_pts = None
    matched_frame_pts = None
    match_vis = None
    detected = False

    if des_frame is not None and len(des_frame) > 0 and ref_des is not None:
        # --- FEATURE MATCHING (Lowe's ratio test) ---
        knn_matches = bf.knnMatch(ref_des, des_frame, k=2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # --- ESTIMASI HOMOGRAPHY & BOUNDING BOX ---
        MIN_MATCH_COUNT = 15
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                dst_pts_projected = cv2.perspectiveTransform(template_pts, M)
                matched_pts = [ref_kp[m.queryIdx].pt for m in good_matches]
                matched_frame_pts = [kp_frame[m.trainIdx].pt for m in good_matches]
                state["last_box"] = dst_pts_projected
                state["last_pts"] = matched_pts
                state["last_frame_pts"] = matched_frame_pts
                state["last_match_vis"] = match_vis
                state["hold"] = HOLD_FRAMES
                detected = True
                if DRAW_BOX:
                    frame_out = cv2.polylines(frame_out, [np.int32(dst_pts_projected)], True, label_color, BOX_THICKNESS, cv2.LINE_AA)
                cv2.putText(
                    frame_out,
                    "Uang Terdeteksi",
                    (int(dst_pts_projected[0][0][0]), int(dst_pts_projected[0][0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    label_color,
                    2,
                )
    if not detected:
        if state["hold"] > 0 and state["last_box"] is not None:
            state["hold"] -= 1
            frame_out = cv2.polylines(frame_out, [np.int32(state["last_box"])], True, label_color, BOX_THICKNESS, cv2.LINE_AA)
            cv2.putText(frame_out, "Tracking...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            matched_pts = state["last_pts"]
            matched_frame_pts = state.get("last_frame_pts")
        else:
            cv2.putText(frame_out, "Mencari Uang...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame_out, filter_name, (20, frame_out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
    return frame_out, matched_pts, matched_frame_pts, match_vis

# ==========================================
# 2. MULAI VIDEO TRACKING (WEBCAM)
# ==========================================
cap = cv2.VideoCapture(0)

state_gauss = {"last_box": None, "last_pts": None, "last_frame_pts": None, "last_match_vis": None, "hold": 0}
state_bilat = {"last_box": None, "last_pts": None, "last_frame_pts": None, "last_match_vis": None, "hold": 0}

print("Kamera aktif. Tekan 'q' pada jendela video untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame dari kamera.")
        break

    # Flip frame secara horizontal agar menjadi cermin (mirror)
    frame = cv2.flip(frame, 1)

    # Proses dua kombinasi filter pada frame yang sama
    frame_gauss, pts_gauss, pts_frame_gauss, match_gauss = track_sift(
        frame,
        "SIFT + Gaussian",
        gray_ref_gauss,
        kp_ref_gauss,
        des_ref_gauss,
        lambda img: cv2.GaussianBlur(img, (5, 5), 0),
        (0, 255, 0),
        state_gauss,
    )
    frame_bilat, pts_bilat, pts_frame_bilat, match_bilat = track_sift(
        frame,
        "SIFT + Bilateral",
        gray_ref_bilat,
        kp_ref_bilat,
        des_ref_bilat,
        lambda img: cv2.bilateralFilter(img, 9, 75, 75),
        (0, 200, 255),
        state_bilat,
    )

    # Tampilkan thumbnail template saja + titik keypoints
    thumb_gauss = template_thumb.copy()
    thumb_bilat = template_thumb.copy()
    if pts_gauss:
        thumb_gauss = draw_match_points_on_thumb(thumb_gauss, pts_gauss, (0, 255, 0))
    if pts_bilat:
        thumb_bilat = draw_match_points_on_thumb(thumb_bilat, pts_bilat, (0, 200, 255))
    frame_gauss = overlay_thumbnail(frame_gauss, thumb_gauss, top=10, left=10)
    frame_bilat = overlay_thumbnail(frame_bilat, thumb_bilat, top=10, left=10)

    if pts_gauss and pts_frame_gauss:
        draw_match_lines_from_thumb(frame_gauss, pts_gauss, pts_frame_gauss, (10, 10), (0, 255, 0))
    if pts_bilat and pts_frame_bilat:
        draw_match_lines_from_thumb(frame_bilat, pts_bilat, pts_frame_bilat, (10, 10), (0, 200, 255))

    # Gabungkan side-by-side agar terlihat dalam satu window
    display_frame = np.hstack([frame_gauss, frame_bilat])

    # Tampilkan video hasil
    cv2.imshow('SIFT Gaussian vs Bilateral', display_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup kamera
cap.release()
cv2.destroyAllWindows()