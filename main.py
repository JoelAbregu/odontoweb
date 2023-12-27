import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

from util import classify, set_background

set_background('./bgs/bg5.png')
# Ocultar la barra de herramientas
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Ocultar la barra negra
st.markdown("""
    <style>
        header {
            display: none;
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# set title
st.title('CARIES - PERIODONTITIS PERIAPICAL')

# set header
st.header('Por favor sube una imagen')
st.markdown('<style>h1, h2, h3, h4, h5, h6 {color: black;}</style>', unsafe_allow_html=True)

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/modelo_clasificador_odonto.h5')

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    # Open and convert image to RGB
    original_image = Image.open(file).convert('RGB')

    # Determine the maximum dimension to maintain aspect ratio
    max_dimension = max(original_image.size)

    # Create a new image with a black background
    new_size = (max_dimension, max_dimension)
    squared_image = Image.new('RGB', new_size, (0, 0, 0))

    # Paste the original image onto the black background
    squared_image.paste(original_image, ((max_dimension - original_image.size[0]) // 2, (max_dimension - original_image.size[1]) // 2))

    # Apply adjustments (brightness, contrast, lights)
    enhanced_image = ImageEnhance.Brightness(squared_image).enhance(0.8)    # -20% brightness
    enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(1.7)      # +70% contrast
    enhanced_image = ImageEnhance.Brightness(enhanced_image).enhance(1.2)   # +20% lights


    # Display the modified image
    st.image(enhanced_image, use_column_width=True)

    # Classify the modified image
    class_name, conf_score = classify(enhanced_image, model, class_names)
    # write classification
    st.write("## {}".format(class_name))
    st.write("### Porcentaje de precisión: {}%".format(int(conf_score * 1000) / 10))




