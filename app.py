import streamlit as st
from PIL import Image

from src.pipeline.prediction_pipeline import PredictionPipeLine

st.title("MNIST Digit Recognition")

uploaded_file = st.file_uploader("Upload a handwritten digit image...",type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Predict"):

        pipeline = PredictionPipeLine()

        result = pipeline.predict(image)

        st.success(f"Predicted digit: {result}")
