# 🌿 Eco-Wise AI Service — Back-End Integration Guide

Dokumentasi ini disusun untuk mempermudah tim _Back-End Developer_ dalam menjalankan, mengonfigurasi, dan mengintegrasikan REST API Klasifikasi Sampah Cerdas Eco-Wise ke dalam arsitektur utama aplikasi.

## 📦 Berkas yang Diserahkan (Artifacts)

Pastikan berkas-berkas berikut berada dalam satu direktori kerja pada server API:

1. `app.py` : Berkas utama layanan REST API menggunakan FastAPI.
2. `inference.py` : Modul mesin inferensi TensorFlow yang menangani pemuatan arsitektur kustom.
3. `model_ecowise_production.keras` : Berkas fisik model _Deep Learning_ siap produksi (MobileNetV2 + Attention).
4. `class_names.json` : Berkas pemetaan indeks kelas ke label tekstual (`Anorganik`, `B3`, `Organik`).
5. `requirements.txt` : Daftar dependensi pustaka Python pihak ketiga.

---

## 🔑 Konfigurasi API Key Gemini (Google AI Studio)

Layanan ini memanfaatkan pustaka `google-genai` terbaru untuk memunculkan fitur edukasi daur ulang sampah.

### Cara Mendapatkan API Key:

1. Masuk ke [Google AI Studio](https://aistudio.google.com/) menggunakan akun Google.
2. Klik tombol **"Create API Key"** di pojok kiri atas.
3. Pilih proyek atau buat proyek baru, lalu salin (_copy_) token API Key yang digenerasikan (berawalan `AIzaSy...`).

### Cara Konfigurasi di Server (Penting untuk Keamanan):

**Jangan menuliskan API Key langsung di dalam kode (hardcode).** Kode pada `app.py` dirancang untuk membaca kunci melalui _Environment Variables_.

- Di lingkungan lokal (Development), buat file `.env` atau set variabel di terminal.
- Di lingkungan Cloud (Production seperti Render/Railway), masukkan kunci tersebut pada menu _Environment Variables Configuration_ dengan nama berikut:

```env
GEMINI_API_KEY=AIzaSyBxxxx_Masukkan_API_Key_Asli_Di_Sini
MODEL_PATH=model_ecowise_production.keras
MAX_FILE_SIZE_MB=10
```
