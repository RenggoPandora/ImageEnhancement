# Image Enhancement (Web)

Aplikasi web interaktif untuk meningkatkan kualitas citra dalam kondisi pencahayaan rendah (low-light image enhancement) menggunakan pipeline pengolahan citra klasik berbasis OpenCV. Aplikasi ini juga mengintegrasikan pengujian berbasis contoh gambar dari dataset LOLv2-real untuk menganalisis akurasi hasil peningkatan dibandingkan dengan citra referensi normal (Ground Truth).

---

## Quick Start

1. Buat Python virtual environment dan aktifkan:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

2. Buka tautan berikut di browser Anda: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## Alur Proses Peningkatan Citra (Enhancement Pipeline)

Algoritma pemrosesan citra dirancang secara deterministik dan berjalan secara adaptif berdasarkan tingkat kegelapan citra. Berikut adalah tahapan proses pengolahan yang dilakukan pada berkas `app.py`:

### 1. Keseimbangan Warna / White Balance (Opsional)
Jika diaktifkan, sistem menerapkan metode **Grayworld White Balance**. Metode ini mengasumsikan rata-rata kontribusi warna merah (R), hijau (G), dan biru (B) pada gambar normal adalah abu-abu homogen. Koreksi dilakukan dengan menghitung rata-rata masing-masing saluran warna dan menyesuaikan faktor pengalinya agar rata-ratanya setara.

### 2. Deteksi Tingkat Kecerahan (Brightness Estimation)
Gambar dikonversi dari ruang warna BGR ke ruang warna **CIELAB** untuk memisahkan informasi kecerahan (saluran L/Luminance) dari informasi warna (saluran a dan b). Rata-rata nilai saluran L dihitung untuk menentukan apakah citra benar-benar tergolong *low-light* (L < 90). Jika gambar sudah terang/normal, sistem akan mempertahankan citra asli tanpa pemrosesan lebih lanjut untuk menghindari distorsi.

### 3. Peredaman Derau Awal (Denoising)
Area gelap sangat rentan terhadap derau sensor (noise). Sebelum kontras ditingkatkan, derau diredam menggunakan metode **Fast Non-Local Means Denoising Colored** (atau **Bilateral Filter** sebagai fallback). Metode ini menyaring derau dengan membandingkan kesamaan piksel di seluruh area sekitar tanpa mengaburkan batas tepi objek secara drastis.

### 4. Peregangan Luminansi (Luminance Stretching)
Peregangan kontras linier diterapkan pada saluran L dengan memetakan nilai piksel dari persentil 2.0 (sebagai batas bawah baru) hingga persentil 98.5 (sebagai batas atas baru) ke rentang penuh 0 - 255. Hal ini memaksimalkan penggunaan rentang dinamis sebelum pemrosesan lokal dilakukan.

### 5. Peningkatan Kontras Lokal (CLAHE)
**Contrast Limited Adaptive Histogram Equalization (CLAHE)** diterapkan pada saluran L. Berbeda dengan ekualisasi histogram biasa yang meratakan seluruh gambar, CLAHE membagi gambar ke dalam grid kecil (8x8 tile) dan meratakan histogramnya secara lokal. Batas klip kontras (*clip limit*) digunakan untuk mencegah penguatan derau yang berlebihan di area yang seragam.

### 6. Penajaman Detail (Unsharp Masking)
Saluran L diperjelas dengan menerapkan teknik **Unsharp Masking**, yaitu menghitung selisih antara saluran L yang tajam dengan versi yang dikaburkan menggunakan Gaussian Blur, lalu menjumlahkannya kembali dengan bobot tertentu untuk menonjolkan tekstur halus.

### 7. Koreksi Gamma Adaptif (Adaptive Gamma Correction)
Untuk mencerahkan bagian gelap secara non-linear tanpa membuat bagian yang sudah terang menjadi terlalu silau (overexposed), diterapkan koreksi Gamma (hukum pangkat / *power law*):
$$L_{out} = L_{in}^{\gamma}$$
Nilai gamma default disetel secara adaptif pada rentang 0.75 untuk citra gelap guna mendongkrak kecerahan secara optimal.

### 8. Penskalan Warna Proporsional (Chroma Scaling)
Meningkatkan kecerahan (L) secara signifikan seringkali membuat warna memudar (*washout*). Untuk mengatasinya, komponen warna `a` dan `b` pada ruang warna CIELAB diskalakan secara proporsional terhadap rasio peningkatan nilai kecerahan $L_{out} / L_{in}$, dibantu dengan parameter *Saturation Boost*.

### 9. Penajaman Selektif Tepi (Edge-Aware Sharpening)
Untuk membatasi agar penajaman tidak memperkuat derau pada area rata (seperti langit malam), dibuat sebuah **masker tepi** menggunakan operator **Laplacian**. Penajaman warna hanya akan diterapkan pada area yang terdeteksi memiliki tepi kuat berdasarkan masker tersebut.

---

## Metodologi Evaluasi Kualitas Citra

Evaluasi kualitas citra hasil peningkatan dilakukan secara objektif dengan membandingkan hasil akhir terhadap citra referensi normal yang ideal (Ground Truth) menggunakan dua metrik standar industri pengolahan citra:

### 1. SSIM (Structural Similarity Index Measure)
SSIM mengukur tingkat kemiripan struktural antara citra hasil peningkatan ($x$) dengan citra Ground Truth ($y$). Tidak seperti MSE yang hanya mengukur perbedaan nilai piksel, SSIM mengevaluasi perubahan persepsi manusia yang mencakup tiga aspek:
* **Luminansi (Kecerahan):** Membandingkan rata-rata intensitas cahaya ($\mu_x$ dan $\mu_y$).
* **Kontras:** Membandingkan simpangan baku/standar deviasi ($\sigma_x$ dan $\sigma_y$).
* **Struktur:** Membandingkan korelasi silang ($\sigma_{xy}$).

Formula matematika SSIM adalah:
$$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

*Di mana $C_1$ dan $C_2$ adalah konstanta penstabil.*
* **Interpretasi Nilai:** Rentang nilai adalah -1 hingga 1 (biasanya dinormalisasi menjadi 0% - 100%). Nilai **100% (1.0)** mengindikasikan kedua citra identik secara sempurna. Nilai akurasi di atas **80%** untuk pemrosesan citra *low-light* sudah dikategorikan sangat memuaskan dan mempertahankan struktur objek dengan sangat baik.

### 2. PSNR (Peak Signal-to-Noise Ratio)
PSNR mengukur rasio antara daya maksimum sinyal gambar terhadap daya derau yang merusaknya. Derau dihitung menggunakan **Mean Squared Error (MSE)** antara citra hasil ($K$) dengan citra referensi ($I$) berukuran $m \times n$:
$$MSE = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}[I(i,j) - K(i,j)]^2$$

Formula PSNR didefinisikan sebagai:
$$PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)$$

*Di mana $MAX_I$ adalah nilai piksel maksimum yang mungkin (255 untuk citra 8-bit).*
* **Interpretasi Nilai:** Diukur dalam satuan desibel (dB). Semakin tinggi nilai PSNR, semakin sedikit derau yang dihasilkan dan semakin mirip citra hasil dengan citra referensi asli. Nilai PSNR di atas **20 dB** umumnya dianggap dapat diterima, sedangkan nilai **25 dB ke atas** dikategorikan sangat baik untuk rekonstruksi citra.

---

## Catatan Teknis Dokumentasi

* Folder dataset `LOLv2-real` harus diletakkan sejajar dengan berkas `app.py`.
* Berkas unggahan kustom disimpan sementara di direktori `static/uploads`, sedangkan hasil pemrosesan diletakkan di `static/results`. Sistem akan menghapus berkas yang berusia lebih dari 1 jam secara otomatis untuk menghemat penyimpanan.
