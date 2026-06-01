# PRODUCT REQUIREMENTS DOCUMENT (PRD)
## Project: Web-Based General Image Enhancement System

---

## 1. Latar Belakang Produk

Kualitas citra digital sering mengalami degradasi akibat noise, pencahayaan yang tidak merata, keterbatasan sensor kamera, serta kondisi lingkungan saat pengambilan gambar. Proses peningkatan kualitas citra secara manual umumnya membutuhkan keahlian teknis khusus dan perangkat lunak pengolahan citra yang tidak selalu mudah diakses oleh pengguna umum.

Dalam konteks akademik, berbagai metode image enhancement telah dikaji dan divalidasi menggunakan dataset benchmark, salah satunya dataset **LOLv2-real**, yang menyediakan pasangan citra low-light dan normal-light. Dataset tersebut memungkinkan evaluasi peningkatan kualitas citra secara objektif dan visual.

Oleh karena itu, diperlukan sebuah sistem berbasis web yang mampu melakukan image enhancement secara otomatis menggunakan teknik pengolahan citra digital dasar, sekaligus memanfaatkan dataset benchmark sebagai **media validasi dan evaluasi hasil enhancement**, sehingga sistem dapat dipertanggungjawabkan secara akademik.

---

## 2. Tujuan Produk

Produk ini bertujuan untuk:
- Meningkatkan kualitas visual citra digital secara otomatis melalui pipeline pengolahan citra yang terdefinisi.
- Mengimplementasikan teori pengolahan citra digital klasik ke dalam sebuah aplikasi berbasis web.
- Menyediakan hasil citra yang lebih jelas dan informatif untuk keperluan visualisasi.
- Menjadi media pembelajaran dan validasi implementasi konsep image enhancement.
- Memanfaatkan dataset benchmark sebagai dasar evaluasi kualitas hasil enhancement.
- Memberikan fasilitas pengunduhan hasil citra kepada pengguna.

---

## 3. Target Pengguna

Target pengguna sistem meliputi:
- Mahasiswa, khususnya pada bidang informatika dan pengolahan citra digital.
- Pengguna umum yang membutuhkan peningkatan kualitas citra secara cepat.
- Dosen atau penguji proyek sebagai pihak evaluasi akademik.

---

## 4. Ruang Lingkup Produk

### 4.1 Termasuk
- Pengunggahan citra digital melalui antarmuka web.
- Proses image enhancement otomatis menggunakan pipeline tetap.
- Visualisasi citra sebelum dan sesudah enhancement.
- Pengunduhan hasil citra enhancement.
- Penggunaan dataset benchmark sebagai data uji dan evaluasi sistem.

### 4.2 Tidak Termasuk
- Pengolahan video.
- Editor citra manual atau interaktif.
- Sistem akun dan autentikasi pengguna.
- Penyimpanan histori citra secara permanen.
- Pelatihan model berbasis machine learning atau deep learning.

---

## 5. Deskripsi Sistem (High-Level)

Sistem merupakan aplikasi berbasis web yang menerima citra digital sebagai input dari pengguna. Citra tersebut diproses menggunakan pipeline pengolahan citra digital klasik yang bersifat deterministik dan statis.

Selain citra input dari pengguna, sistem juga memanfaatkan dataset **LOLv2-real** sebagai:
- Data uji internal untuk pengujian fungsional pipeline.
- Acuan visual dalam mengevaluasi kualitas hasil enhancement.
- Bahan dokumentasi dan analisis hasil pada laporan akademik.

Hasil pemrosesan ditampilkan dalam bentuk perbandingan citra asli dan citra hasil enhancement, serta dapat diunduh oleh pengguna.

---

## 6. Alur Kerja Sistem

Alur kerja sistem secara umum adalah sebagai berikut:
1. Pengguna membuka halaman web sistem.
2. Pengguna mengunggah citra digital atau memilih citra contoh dari dataset.
3. Sistem melakukan validasi format dan ukuran file.
4. Sistem menjalankan pipeline image enhancement.
5. Sistem menampilkan citra asli dan citra hasil enhancement.
6. Pengguna mengunduh citra hasil enhancement.

---

## 7. Pemanfaatan Dataset

### 7.1 Dataset yang Digunakan
Dataset yang digunakan dalam pengembangan dan evaluasi sistem adalah **LOLv2-real**.

### 7.2 Struktur Dataset
Dataset memiliki struktur sebagai berikut:

- `train/low` : citra low-light untuk pengujian pipeline
- `train/high` : citra referensi normal-light
- `test/low` : citra low-light untuk evaluasi
- `test/high` : citra ground truth
- `metadata.json` : informasi pasangan citra

### 7.3 Fungsi Dataset dalam Sistem
Dataset digunakan untuk:
- Menguji efektivitas pipeline image enhancement.
- Membandingkan hasil enhancement dengan citra normal-light.
- Menyediakan contoh citra uji yang konsisten dan terstandarisasi.
- Mendukung analisis visual dan pembahasan hasil pada laporan akhir.

Dataset **tidak digunakan untuk pelatihan model**, melainkan sebagai **data evaluasi dan validasi metode klasik**.

---

## 8. Fitur Utama (Functional Requirements)

### FR-01 Upload Citra
- Sistem menerima file citra dengan format JPG dan PNG.
- Ukuran maksimum file dibatasi (misalnya ≤ 5 MB).

### FR-02 Validasi File
- Sistem menolak file yang bukan merupakan citra.
- Sistem menampilkan pesan kesalahan apabila proses unggah gagal.

### FR-03 Image Enhancement Pipeline
- Sistem menerapkan pipeline pengolahan citra yang bersifat tetap dan berurutan:
  - Konversi warna (RGB ke Grayscale bila diperlukan).
  - Reduksi noise menggunakan filter smoothing.
  - Penajaman citra (sharpening).
  - Penyesuaian kontras atau intensitas.
- Metode yang digunakan merupakan metode pengolahan citra klasik.

### FR-04 Visualisasi Output
- Sistem menampilkan citra asli dan citra hasil enhancement secara berdampingan.
- Perbandingan visual dibuat sederhana dan mudah dipahami.

### FR-05 Unduh Hasil Citra
- Sistem menyediakan tombol unduh hasil citra.
- Format file keluaran berupa JPG atau PNG.
- Nama file hasil dihasilkan secara otomatis.

---

## 9. Kebutuhan Non-Fungsional (Non-Functional Requirements)

- **Usability**: Antarmuka sederhana dan intuitif.
- **Performance**: Waktu pemrosesan beberapa detik per citra.
- **Reliability**: Sistem stabil untuk berbagai citra umum.
- **Portability**: Berjalan pada browser modern tanpa plugin tambahan.
- **Data Handling**: File disimpan sementara dan dihapus otomatis setelah sesi selesai.

---

## 10. Teknologi yang Digunakan

- Frontend: HTML, CSS
- Backend: Python
- Framework: Flask
- Library pengolahan citra: OpenCV dan/atau scikit-image

---

## 11. Batasan Teknis

- Parameter enhancement bersifat statis.
- Tidak menjamin hasil optimal untuk semua jenis citra.
- Tidak mendukung pemrosesan citra real-time.
- Tidak menggunakan metode berbasis AI atau deep learning.

---

## 12. Kriteria Keberhasilan

Sistem dianggap berhasil apabila:
- Sistem dapat memproses citra tanpa error.
- Hasil enhancement menunjukkan peningkatan kualitas visual.
- Pipeline dapat dijelaskan secara logis dan akademik.
- Dataset dapat digunakan untuk evaluasi hasil secara konsisten.
- Pengguna berhasil mengunduh hasil citra.

---

## 13. Risiko dan Mitigasi

- **Risiko**: Hasil enhancement kurang signifikan pada citra tertentu.  
  **Mitigasi**: Menggunakan metode enhancement klasik yang stabil dan umum.

- **Risiko**: File input tidak valid.  
  **Mitigasi**: Validasi format dan ukuran file secara ketat.