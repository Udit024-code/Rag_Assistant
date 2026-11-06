import os
from dotenv import load_dotenv

# Load environment variables from .env file (used when running locally)
load_dotenv()


def get_secret(name):
    """Read an API key from the environment (.env locally) or, when deployed
    on Streamlit Cloud, from st.secrets. Returns None if it isn't set."""
    value = os.getenv(name)
    if not value:
        try:
            import streamlit as st
            value = st.secrets[name]
        except Exception:
            pass
    return value
