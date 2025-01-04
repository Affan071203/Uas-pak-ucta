import cv2
import numpy as np

# Fungsi untuk deteksi warna
def detect_color(image, lower_bound, upper_bound, color_label):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter area kecil
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Warna hijau untuk bounding box
            cv2.putText(image, color_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return image

# Fungsi untuk deteksi wajah
def detect_faces(image, face_cascade):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Warna biru untuk bounding box wajah
    return image

# Fungsi utama
def main():
    # Memuat gambar
    image_path = "2.jpg"  # Ganti dengan path gambar Anda
    image = cv2.imread(image_path)
    
    if image is None:
        print("Gambar tidak ditemukan!")
        return

    # Resize gambar agar lebih kecil
    scale_percent = 100  # Persentase ukuran baru
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(image, (width, height))
    print(f"Gambar diubah ukurannya menjadi: {width}x{height}")
    
    # Deteksi Warna Hijau
    lower_green = np.array([25, 40, 40])  # Rentang bawah hijau dalam HSV
    upper_green = np.array([80, 255, 255])  # Rentang atas hijau dalam HSV
    color_detected_image_green = detect_color(resized_image.copy(), lower_green, upper_green, "Green")

    # Deteksi Warna Putih
    lower_white = np.array([0, 0, 200])  # Rentang bawah putih dalam HSV
    upper_white = np.array([180, 30, 255])  # Rentang atas putih dalam HSV
    color_detected_image_white = detect_color(resized_image.copy(), lower_white, upper_white, "White")

    # Deteksi Wajah
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_detected_image = detect_faces(resized_image.copy(), face_cascade)
    
    # Konversi ke Grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray_image_path = "grayscale_image.jpg"
    cv2.imwrite(gray_image_path, gray_image)
    print(f"Gambar grayscale disimpan di: {gray_image_path}")
    
    # Menampilkan gambar
    cv2.imshow("Gambar Asli", resized_image)
    cv2.imshow("Deteksi Warna Hijau", color_detected_image_green)
    cv2.imshow("Deteksi Warna Putih", color_detected_image_white)
    cv2.imshow("Deteksi Wajah", face_detected_image)
    cv2.imshow("Grayscale", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
