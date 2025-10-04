# fake_profile_streamlit_fixed.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PIL import Image
import os

# ----------------- Load dataset -----------------
dataset_path = os.path.join(os.path.dirname(__file__), "Dataset.csv")
df = pd.read_csv(dataset_path)

X = df.drop('fake', axis=1)
y = df['fake']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ----------------- Streamlit UI -----------------
st.title("Fake Profile Detection")
st.write("Enter the profile information. If True enter 1 or Enter 0")

profile = st.selectbox("Profile", [0, 1])
username = st.selectbox("Username", [0, 1])
URL = st.selectbox("Link", [0, 1])
posts = st.number_input("Posts", min_value=0)
followers = st.number_input("Followers", min_value=0)
follows = st.number_input("Following", min_value=0)

if st.button("Check the profile"):
    profile_data = [profile, username, URL, posts, followers, follows]
    prediction = model.predict([profile_data])[0]
    result = "Real" if prediction == 0 else "Fake"

    with st.expander("", expanded=True):
        st.write(f"Profile is **{result}**")

        # ----------------- Load image from project folder -----------------
        if result == "Fake":
            img_path = os.path.join(os.path.dirname(__file__), "Not ok.png")
        else:
            img_path = os.path.join(os.path.dirname(__file__), "Ok Emoji.png")

        img = Image.open(img_path)
        st.image(img, width=200)
