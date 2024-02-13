import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
 
model = tf.keras.models.load_model('cnn_attention.h5')

# Custom CSS for dark mode and background image
custom_css = """
    <style>
        body {
            color: white;
            background-color: #2E4053;
            background-image: url('https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Aa35cz76rGh6PiDb2UEs8w.jpeg');  /* Replace with your image URL */
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }
        .sidebar .sidebar-content {
            background-color: #34495E;
        }
        .sidebar .sidebar-content .stRadio label, .sidebar .sidebar-content .stRadio span {
            color: white;
        }
        .sidebar .sidebar-content .stRadio [class^="stRadio"] > div {
            background-color: #34495E;
        }
    </style>
"""
 
st.set_page_config(
    page_title="Malaria Detection App",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",  # Expanded sidebar by default
)
st.markdown(custom_css, unsafe_allow_html=True)


def load_image(image_file):
    img = Image.open(image_file)
    return img

def preprocess_input_image(img):
    # Resize the image to (64, 64, 3)
    img = img.resize((64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

def predict_malaria(img_array): 
    predictions = model.predict(img_array)
    return predictions[0][0]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Malaria Information", "Detect Malaria"])


if page == "Malaria Information":
    st.title("Malaria Information")
    st.write("""
    ## What is Malaria?

    Malaria is a life-threatening mosquito-borne blood disease caused by a Plasmodium parasite. It is transmitted to humans through the bite of the female Anopheles mosquito.

    ### Symptoms of Malaria

    - Fever
    - Chills
    - Sweats
    - Headaches
    - Nausea and vomiting
    - Body aches
    - Fatigue

    ### Prevention

    - Use insect repellent.
    - Sleep under a mosquito net.
    - Take antimalarial drugs if recommended.

    ### Treatment

    Malaria is treatable with antimalarial medications, but prompt diagnosis and treatment are crucial.
    """)

elif page == "Detect Malaria":
    st.title("Detect Malaria")
    st.write("Upload a blood smear image to check for malaria infection.")
 
    uploaded_file = st.file_uploader("Choose a blood smear image...", type=["jpg", "jpeg", "png"])
 
    if uploaded_file is not None: 
        original_image = load_image(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)
        img_array = preprocess_input_image(original_image)
        prediction = predict_malaria(img_array)
 
        st.subheader("Prediction:")
        if prediction > 0.5:
            st.error("Infected with Malaria 🦠")
        else:
            st.success("Not infected with Malaria 🎉")
 
        # st.subheader("Probability:")
        # st.write(f"{prediction:.2%}")
 
st.markdown("---")
st.write("Built with ❤️ by Naveen Kumar Kateghar")
