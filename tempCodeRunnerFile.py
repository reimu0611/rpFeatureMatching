import cv2
import numpy as np

# ==========================================
# 1. INISIALISASI & PERSIAPAN TEMPLATE
# ==========================================
# Inisialisasi detektor ORB. 
# nfeatures=1000 agar lebih banyak titik sudut yang terdeteksi
orb = cv2.ORB_create(nfeatures=1000)

# Inisialisasi Brute-Force Matcher dengan NORM_HAMMING (Wajib untuk ORB/Binary Descriptor)
# crossCheck=True akan otomatis menyaring kecocokan bolak-balik yang paling valid
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Load gambar template referensi
img_ref = cv2.imread('template-uang.png')

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
# ... (kode selanjutnya tetap sama)

if img_ref is None:
    print("Error: Gambar template_uang.jpg tidak ditemukan! Pastikan nama dan lokasi file benar.")
    exit()

# --- PREPROCESSING TEMPLATE (Materi Dosen) ---
# 1. Grayscale
gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
# 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_ref = clahe.apply(gray_ref)

# Ekstrak Keypoints (kp) dan Descriptors (des) dari Template
kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)

# Ambil dimensi template untuk menggambar Bounding Box nanti
h, w = gray_ref.shape
template_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

# ==========================================
# 2. MULAI VIDEO TRACKING (WEBCAM)
# ==========================================
cap = cv2.VideoCapture(0)

print("Kamera aktif. Tekan 'q' pada jendela video untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame dari kamera.")
        break

    # Flip frame secara horizontal agar menjadi cermin (mirror)
    frame = cv2.flip(frame, 1)

    # --- PREPROCESSING FRAME KAMERA ---
    # 1. Grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. CLAHE untuk mengatasi pantulan cahaya / redup
    gray_frame = clahe.apply(gray_frame)
    # 3. Gaussian Blur ringan untuk mengurangi noise sensor kamera
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # Ekstrak fitur dari frame kamera saat ini
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    # Pastikan descriptor ditemukan sebelum melakukan pencocokan
    if des_frame is not None and len(des_frame) > 0:
        
        # --- FEATURE MATCHING ---
        matches = bf.match(des_ref, des_frame)
        
        # Urutkan berdasarkan jarak (distance). Jarak terkecil = paling mirip
        matches = sorted(matches, key=lambda x: x.distance)

        # Ambil 15% kecocokan terbaik (Good Matches) untuk stabilitas
        good_matches = matches[:int(len(matches) * 0.15)]

        # --- ESTIMASI HOMOGRAPHY & BOUNDING BOX ---
        # Tentukan batas minimal titik kecocokan (misal: butuh minimal 15 titik agar diakui sebagai uang)
        MIN_MATCH_COUNT = 15 
        matchesMask = None

        if len(good_matches) > MIN_MATCH_COUNT:
            # Ambil koordinat (x, y) dari titik-titik yang cocok
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Hitung matriks Homography menggunakan RANSAC untuk membuang outlier (titik salah)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Jika matriks ditemukan, gambar kotak pelacakan
            if M is not None:
                matchesMask = mask.ravel().tolist() # Simpan titik-titik valid (inliers)
                # Proyeksikan 4 titik sudut template ke posisi uang di kamera
                dst_pts_projected = cv2.perspectiveTransform(template_pts, M)

                # Gambar kotak hijau (Bounding Box) di sekeliling uang
                frame = cv2.polylines(frame, [np.int32(dst_pts_projected)], True, (0, 255, 0), 3, cv2.LINE_AA)
                
                # Tambahkan label teks
                cv2.putText(frame, "Uang Terdeteksi", (int(dst_pts_projected[0][0][0]), int(dst_pts_projected[0][0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Mencari Uang...", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if matchesMask is None:
            matchesMask = [0] * len(good_matches) # Sembunyikan garis jika tidak valid

        # --- VISUALISASI TITIK (Aktifkan untuk melihat garis pencocokan dan template) ---
        display_frame = cv2.drawMatches(img_ref, kp_ref, frame, kp_frame, good_matches, None, matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        display_frame = frame

    # Tampilkan video hasil
    cv2.imshow('ORB Real-Time Rupiah Tracker', display_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup kamera
cap.release()
cv2.destroyAllWindows()