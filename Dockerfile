# Menggunakan image dasar Python 3.10 versi slim agar ukuran image lebih kecil
FROM python:3.10-slim

# Mencegah Python membuat file .pyc (bytecode)
ENV PYTHONDONTWRITEBYTECODE=1
# Memastikan output console langsung ditampilkan tanpa buffer
ENV PYTHONUNBUFFERED=1

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Menyalin file requirements.txt terlebih dahulu
COPY requirements.txt .

# Menginstal dependensi sistem (jika diperlukan oleh TensorFlow/Pillow) dan dependensi Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Menyalin seluruh kode proyek ke dalam direktori kerja container
COPY . .

# Mengekspos port 8000 agar dapat diakses dari luar container
EXPOSE 8000

# Perintah untuk menjalankan aplikasi FastAPI menggunakan Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
