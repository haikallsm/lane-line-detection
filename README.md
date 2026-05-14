# Lane Line Detection System

Sistem deteksi garis lajur kendaraan berbasis computer vision menggunakan OpenCV, berdasarkan metodologi jurnal:

**"Lane Line Detection and Object Scene Segmentation Using Otsu Thresholding and the Fast Hough Transform for Intelligent Vehicles in Complex Road Conditions"**

## Persyaratan Sistem

- **Python 3.10+**

## Instalasi

### 1. Setup Virtual Environment
```bash
# Buat folder project
mkdir lane-detection
cd lane-detection

# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate

# Install Dependencies
pip install --upgrade pip
pip install opencv-python numpy matplotlib pathlib

# Run Program
python lane_detect.py

#Pilih nomor 2 lalu isi dengan
<nama_video>.mp4
