import streamlit as st
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix
import pandas as pd

st.set_page_config(layout="wide")
st.title("Ekstraksi Matriks Tekstur")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Tampilkan Gambar yang Di-upload
    st.subheader("🖼️ Gambar yang Di-upload")
    img = Image.open(uploaded_file)
    st.image(img, width=300)
    
    # Convert ke grayscale untuk proses matriks
    img_gray = img.convert('L')
    img_array = np.array(img_gray)
    
    # --- PROSES METODE ---

    # 1. MATRIKS LBP
    st.subheader("1. Matriks Local Binary Pattern (LBP)")
    lbp = local_binary_pattern(img_array, P=8, R=1, method='default')
    # Menggunakan dataframe agar rapi dan bisa di-scroll
    st.dataframe(pd.DataFrame(lbp).astype(int)) 

    # 2. MATRIKS GLCM
    st.subheader("2. Matriks GLCM (Co-occurrence)")
    glcm = graycomatrix(img_array, distances=[1], angles=[0], levels=256)
    glcm_2d = glcm[:, :, 0, 0]
    st.dataframe(pd.DataFrame(glcm_2d))

    # 3. MATRIKS FOURIER TRANSFORM
    st.subheader("3. Matriks Fourier Transform (Magnitude)")
    f_transform = np.fft.fft2(img_array)
    f_shift = np.fft.fftshift(f_transform)
    # Kita tampilkan Magnitudenya (Absolut) agar angkanya tidak terlalu kompleks
    magnitude = np.abs(f_shift)
    st.dataframe(pd.DataFrame(magnitude))

    # --- PENJELASAN ---
    st.info("""
    **Kenapa tampilannya dipotong?** Secara default, jika gambar Anda berukuran 512x512, maka ada 262.144 angka. 
    Streamlit menggunakan tabel interaktif (Dataframe) di atas agar Anda bisa **scroll ke kanan dan ke bawah** untuk melihat nilainya tanpa merusak tampilan layar.
    """)