import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.set_page_config(page_title="Clasificador de Imágenes", page_icon="🧠")

st.title("Clasificador de Imágenes - CNN")
st.caption("Desarrollado por [Tu Nombre] - UTH - Computación en la Nube")

clases = ['camiseta', 'pantalon', 'sueter', 'vestido', 'abrigo',
          'sandalia', 'camisa', 'tenis', 'bolso', 'bota']

@st.cache_resource
def cargar_modelo():
    return load_model('modelo_fashion.h5')

modelo = cargar_modelo()

st.write("Sube una imagen o toma una foto para identificar el objeto.")

opcion = st.radio("Elige una opción:", ["Subir imagen", "Tomar foto"])

img_file = None
if opcion == "Subir imagen":
    img_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
else:
    img_file = st.camera_input("Toma una foto")

if img_file is not None:
    img = Image.open(img_file).convert("L")
    st.image(img, caption="Imagen ingresada", use_container_width=True)

    img_resized = img.resize((28, 28))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    pred = modelo.predict(img_array)
    clase_predicha = clases[np.argmax(pred)]
    confianza = float(np.max(pred))

    st.subheader(f"Predicción: {clase_predicha}")
    st.write(f"Confianza: {confianza:.2f}")

    with st.expander("Ver todas las probabilidades"):
        for i, c in enumerate(clases):
            st.write(f"{c}: {pred[0][i]:.2f}")
else:
    st.info("Esperando una imagen para analizar...")
