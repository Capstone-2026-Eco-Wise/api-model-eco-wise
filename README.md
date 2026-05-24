# Cara Menjalankan API EcoWise Waste Classifier

## Menjalankan secara Lokal

1. **Buka folder proyek**:
   ```bash
   cd api-model-eco-wise
   ```

2. **Buat virtual environment** (disarankan):
   ```bash
   python -m venv venv
   # Di Windows:
   venv\Scripts\activate
   # Di Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi Environment Variables**:
   Salin `.env.example` menjadi `.env` dan sesuaikan isinya jika diperlukan. Pastikan path model dan label class sudah sesuai.
   ```bash
   cp .env.example .env
   ```

5. **Jalankan API**:
   Gunakan Uvicorn untuk menjalankan server FastAPI.
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```
   API dapat diakses di `http://localhost:8000`. Dokumentasi interaktif (Swagger UI) tersedia di `http://localhost:8000/docs`.

---

## Menjalankan dengan Docker

1. Pastikan Docker dan Docker Compose sudah terinstall di sistem Anda.
2. Jalankan perintah berikut di dalam folder `api-model-eco-wise` untuk mem-build dan menjalankan container:
   ```bash
   docker-compose up -d --build
   ```
3. API akan berjalan secara background dan dapat diakses di `http://localhost:8000`.
