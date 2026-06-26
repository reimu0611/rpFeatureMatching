import cv2
import numpy as np
import time

def main():
    # --- 1. PATH GAMBAR REFERENSI ---
    ref_image_path = 'image/template-uang.png'

    # --- 2. INISIALISASI ORB & MATCHER ---
    orb = cv2.ORB_create(nfeatures=500)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # --- 3. PROSES GAMBAR REFERENSI ---
    ref_img_color = cv2.imread(ref_image_path)
    if ref_img_color is None:
        print(f"Error: Tidak dapat menemukan file referensi di '{ref_image_path}'")
        return

    CAM_W, CAM_H = 640, 480

    # Putar gambar menjadi horizontal jika orientasinya vertikal (tinggi > lebar)
    tinggi_asli, lebar_asli = ref_img_color.shape[:2]
    if tinggi_asli > lebar_asli:
        ref_img_color = cv2.rotate(ref_img_color, cv2.ROTATE_90_COUNTERCLOCKWISE)
        tinggi_asli, lebar_asli = ref_img_color.shape[:2]
    
    # Mengecilkan gambar menjadi lebar 400 piksel agar proporsional sebagai referensi
    rasio = 400 / lebar_asli
    REF_W = 400
    REF_H = int(tinggi_asli * rasio)
    ref_img_color = cv2.resize(ref_img_color, (REF_W, REF_H), interpolation=cv2.INTER_AREA)

    ref_gray        = cv2.cvtColor(ref_img_color, cv2.COLOR_BGR2GRAY)
    kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)

    if des_ref is None:
        print("Error: Tidak ada fitur yang ditemukan pada gambar referensi.")
        return

    # --- 4. AKTIFKAN KAMERA ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    print("Kamera aktif. Arahkan kamera ke objek referensi.")
    print("Tekan 'q' untuk keluar.")

    font       = cv2.FONT_HERSHEY_SIMPLEX
    COLOR_G    = (0, 255, 0)
    COLOR_Y    = (0, 255, 255)
    COLOR_O    = (0, 165, 255)
    draw_flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil frame.")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        frame = cv2.resize(frame, (CAM_W, CAM_H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- 5. FILTERING ---
        gaussian_filtered  = cv2.GaussianBlur(gray, (15, 15), 0)
        bilateral_filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # --- 6. DETEKSI & MATCHING ---
        kp_gauss, des_gauss = orb.detectAndCompute(gaussian_filtered, None)
        good_gauss = []
        inlier_ratio_gauss = 0.0
        if des_gauss is not None:
            m = bf.match(des_ref, des_gauss)
            m = sorted(m, key=lambda x: x.distance)
            good_gauss = m[:50]
            if len(good_gauss) > 10:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_gauss]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_gauss[m.trainIdx].pt for m in good_gauss]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    inliers = np.sum(mask)
                    inlier_ratio_gauss = (inliers / len(good_gauss)) * 100

        kp_bilateral, des_bilateral = orb.detectAndCompute(bilateral_filtered, None)
        good_bilateral = []
        inlier_ratio_bilateral = 0.0
        if des_bilateral is not None:
            m = bf.match(des_ref, des_bilateral)
            m = sorted(m, key=lambda x: x.distance)
            good_bilateral = m[:50]
            if len(good_bilateral) > 10:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_bilateral]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_bilateral[m.trainIdx].pt for m in good_bilateral]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    inliers = np.sum(mask)
                    inlier_ratio_bilateral = (inliers / len(good_bilateral)) * 100

        # --- 7. BUAT PANEL MATCHING (ref kecil kiri | cam kanan) ---
        match_gauss = cv2.drawMatches(
            ref_img_color, kp_ref,
            frame, kp_gauss,
            good_gauss, None,
            matchColor=COLOR_G, flags=draw_flags
        )
        match_bilateral = cv2.drawMatches(
            ref_img_color, kp_ref,
            frame, kp_bilateral,
            good_bilateral, None,
            matchColor=COLOR_G, flags=draw_flags
        )

        # --- 8. LABEL ---
        cv2.putText(match_gauss,
                    f"ORB + Gaussian",
                    (10, CAM_H - 15), font, 0.65, COLOR_Y, 2, cv2.LINE_AA)
        
        jml_kp_gauss = len(kp_gauss) if kp_gauss else 0
        cv2.putText(match_gauss, f"Keypoints: {jml_kp_gauss}", (20, REF_H + 30), font, 0.6, (255,255,255), 2)
        cv2.putText(match_gauss, f"Good Matches: {len(good_gauss)}", (20, REF_H + 60), font, 0.6, (255,255,255), 2)
        cv2.putText(match_gauss, f"Inlier Ratio: {inlier_ratio_gauss:.1f}%", (20, REF_H + 90), font, 0.6, (255,255,255), 2)

        cv2.putText(match_bilateral,
                    f"ORB + Bilateral",
                    (10, CAM_H - 15), font, 0.65, COLOR_O, 2, cv2.LINE_AA)

        jml_kp_bilat = len(kp_bilateral) if kp_bilateral else 0
        cv2.putText(match_bilateral, f"Keypoints: {jml_kp_bilat}", (20, REF_H + 30), font, 0.6, (255,255,255), 2)
        cv2.putText(match_bilateral, f"Good Matches: {len(good_bilateral)}", (20, REF_H + 60), font, 0.6, (255,255,255), 2)
        cv2.putText(match_bilateral, f"Inlier Ratio: {inlier_ratio_bilateral:.1f}%", (20, REF_H + 90), font, 0.6, (255,255,255), 2)

        # --- 9. GABUNG HORIZONTAL ---
        gabungan = cv2.hconcat([match_gauss, match_bilateral])
        
        # Tampilkan FPS di ujung kiri atas gambar gabungan
        cv2.putText(gabungan, f"FPS: {int(fps)}", (20, 40), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Scale down agar muat di layar
        h, w = gabungan.shape[:2]
        scale = min(1.0, 1400 / w, 700 / h)
        if scale < 1.0:
            gabungan = cv2.resize(gabungan, (int(w * scale), int(h * scale)))

        cv2.imshow('ORB Matching: Gaussian vs Bilateral', gabungan)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()