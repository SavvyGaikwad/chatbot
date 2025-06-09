try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    pass

import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
import requests
from io import BytesIO
import os

# Setup
genai.configure(api_key="AIzaSyAvzloY_NyX-yjtZb8EE_RdXPs3rPmMEso")  # Replace with your actual API key
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="vector_db", embedding_function=embedding_model)

# Utilities
def convert_github_url_to_raw(github_url):
    if "github.com" in github_url and "/blob/" in github_url:
        return github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return github_url

def is_gif_file(url):
    """Check if the URL points to a GIF file"""
    return url.lower().endswith('.gif')

def display_image_from_url(image_url):
    try:
        raw_url = convert_github_url_to_raw(image_url)
        
        # Handle GIFs differently
        if is_gif_file(raw_url):
            # Method 1: Display GIF using HTML (recommended)
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="{raw_url}" alt="{os.path.basename(image_url)}" style="max-width: 100%; height: auto;">
                <p><em>{os.path.basename(image_url)}</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Alternative Method 2: Show download link for full GIF
            st.markdown(f"🎬 [View full animated GIF]({raw_url})")
            
        else:
            # Handle regular images as before
            response = requests.get(raw_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=os.path.basename(image_url), use_column_width=True)
        
        return True
    except Exception as e:
        st.error(f"❌ Failed to display image: {e}")
        return False

def display_gif_with_frames(image_url):
    """Alternative method to display GIF frames individually"""
    try:
        raw_url = convert_github_url_to_raw(image_url)
        response = requests.get(raw_url, timeout=10)
        response.raise_for_status()
        
        # Open the GIF
        gif = Image.open(BytesIO(response.content))
        
        # Check if it's animated
        if hasattr(gif, 'n_frames') and gif.n_frames > 1:
            st.write(f"🎬 Animated GIF with {gif.n_frames} frames:")
            
            # Create columns for frame display
            cols = st.columns(min(4, gif.n_frames))
            
            # Display first few frames
            for i in range(min(4, gif.n_frames)):
                gif.seek(i)
                frame = gif.copy()
                with cols[i]:
                    st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
            
            # Show link to full GIF
            st.markdown(f"[🔗 View complete animated GIF]({raw_url})")
        else:
            # Single frame image
            st.image(gif, caption=os.path.basename(image_url), use_column_width=True)
            
        return True
    except Exception as e:
        st.error(f"❌ Failed to display GIF frames: {e}")
        return False

def collect_images_from_docs(docs):
    all_images = []
    seen_images = set()
    for doc in docs:
        if hasattr(doc, 'metadata') and doc.metadata:
            images_str = doc.metadata.get('images', '')
            if images_str.strip():
                images = [img.strip() for img in images_str.split('|') if img.strip()]
                for img in images:
                    if img and img not in seen_images:
                        all_images.append(img)
                        seen_images.add(img)
    return all_images

def ask_question_streamlit(query, show_images=True, gif_display_method="html"):
    try:
        docs = vector_db.similarity_search(query, k=5)
        context_parts = [doc.page_content for doc in docs]
        context = "\n\n---\n\n".join(context_parts)

        image_urls = collect_images_from_docs(docs) if show_images else []

        prompt = f"""
You are a helpful assistant answering questions about user profile settings and preferences.

Use the provided context to answer the question accurately and completely. If the context contains relevant information, provide a detailed answer. If you're not sure or the context doesn't contain enough information, say so honestly.

Context:
{context}

Question: {query}

Please provide a clear, helpful answer based on the context above:
"""

        response = model.generate_content(prompt)
        st.markdown("### 🧠 Chatbot Answer")
        st.write(response.text)

        if show_images and image_urls:
            st.markdown("### 📸 Related Images")
            
            for img_url in image_urls:
                if is_gif_file(img_url):
                    st.markdown(f"**🎬 {os.path.basename(img_url)}** (Animated GIF)")
                    
                    if gif_display_method == "html":
                        display_image_from_url(img_url)
                    elif gif_display_method == "frames":
                        display_gif_with_frames(img_url)
                    else:
                        display_image_from_url(img_url)
                else:
                    display_image_from_url(img_url)

        with st.expander("🔍 Debug Info"):
            st.write(f"Used {len(docs)} documents for context.")
            st.write(f"Found {len(image_urls)} related images.")
            gif_count = sum(1 for img in image_urls if is_gif_file(img))
            if gif_count > 0:
                st.write(f"Found {gif_count} animated GIFs.")

    except Exception as e:
        st.error(f"Error occurred: {e}")

# Streamlit UI
st.set_page_config(page_title="Gemini QA Chatbot with Images", layout="wide")
st.title("Gemini QA Chatbot with Image Support")
st.markdown("Ask questions and see image references including animated GIFs!")

# Sidebar for GIF display options
with st.sidebar:
    st.header("GIF Display Options")
    gif_method = st.radio(
        "How to display GIFs:",
        ["html", "frames"],
        format_func=lambda x: "HTML Embed (Animated)" if x == "html" else "Show Frames"
    )
    
    st.info("""
    **HTML Embed**: Shows the full animated GIF
    **Show Frames**: Displays individual frames + link to full GIF
    """)

query = st.text_input("Enter your question:")
show_images = st.checkbox("Show Related Images", value=True)

if st.button("Submit"):
    if query.strip():
        ask_question_streamlit(query, show_images, gif_method)
    else:
        st.warning("Please enter a valid question.")

# Add some example queries
with st.expander("💡 Try these example questions"):
    example_queries = [
        "How do I reset my password?",
        "How to change password?",
        "Show me mobile device settings",
        "How to access my profile?",
        "What are the password requirements?"
    ]
    
    for i, example in enumerate(example_queries):
        if st.button(f"📝 {example}", key=f"example_{i}"):
            ask_question_streamlit(example, True, gif_method)
