import streamlit as st
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix
import pandas as pd

st.set_page_config(page_title="Ekstraksi Fitur Tekstur", layout="wide")

st.title("🔍 Analisis Tekstur (Sampel Matriks)")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan Gambar Asli
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diupload", width=250)
    
    # Convert ke Grayscale
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    h, w = img_array.shape

    # --- 1. MATRIX LBP (Sampel 10x10 pojok kiri atas) ---
    st.subheader("1. MATRIX LBP (Sampel 10x10 pojok kiri atas)")
    lbp = local_binary_pattern(img_array, P=8, R=1, method='default')
    # Ambil irisan 10 baris pertama dan 10 kolom pertama
    lbp_sample = lbp[0:10, 0:10]
    st.dataframe(pd.DataFrame(lbp_sample).astype(int))

    # --- 2. MATRIX GLCM (Sampel intensitas 0-10) ---
    st.subheader("2. MATRIX GLCM (Sampel intensitas 0-10)")
    # Kita gunakan levels=256 tapi hanya tampilkan irisan kecil matriksnya
    glcm = graycomatrix(img_array, distances=[1], angles=[0], levels=256)
    glcm_2d = glcm[:, :, 0, 0]
    # Ambil irisan baris 0-10 dan kolom 0-10
    glcm_sample = glcm_2d[0:11, 0:11]
    st.dataframe(pd.DataFrame(glcm_sample))

    # --- 3. MATRIX FOURIER (Sampel 10x10 pusat frekuensi) ---
    st.subheader("3. MATRIX FOURIER (Sampel 10x10 pusat frekuensi)")
    f_transform = np.fft.fft2(img_array)
    f_shift = np.fft.fftshift(f_transform)
    
    # Mencari titik tengah matriks
    cy, cx = h // 2, w // 2
    # Ambil 10x10 di sekitar pusat (frekuensi rendah)
    fourier_sample = np.abs(f_shift[cy-5:cy+5, cx-5:cx+5])
    st.dataframe(pd.DataFrame(fourier_sample))

    st.success("Matriks berhasil dilimit untuk tampilan yang lebih rapi!")
