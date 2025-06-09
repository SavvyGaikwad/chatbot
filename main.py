import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import webbrowser
import os

# Configure Gemini
genai.configure(api_key="AIzaSyAvzloY_NyX-yjtZb8EE_RdXPs3rPmMEso")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Load Vector DB
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="vector_db", embedding_function=embedding_model)

def convert_github_url_to_raw(github_url):
    """Convert GitHub blob URL to raw URL for direct image access"""
    if "github.com" in github_url and "/blob/" in github_url:
        return github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return github_url

def display_image_from_url(image_url, max_width=600):
    """Display image from URL using matplotlib"""
    try:
        # Convert GitHub URL to raw URL
        raw_url = convert_github_url_to_raw(image_url)
        
        # Download and display image
        response = requests.get(raw_url, timeout=10)
        response.raise_for_status()
        
        # Load image
        img = Image.open(BytesIO(response.content))
        
        # Create matplotlib figure
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Reference Image")
        plt.tight_layout()
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Could not display image from {image_url}")
        print(f"   Error: {str(e)}")
        return False

def display_images_html(image_urls, max_width=400):
    """Create HTML file to display images in browser"""
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Related Images</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ display: flex; flex-wrap: wrap; gap: 15px; }}
                .image-card {{ 
                    border: 1px solid #ccc; 
                    padding: 10px; 
                    border-radius: 8px; 
                    max-width: {max_width}px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .image-card img {{ 
                    max-width: 100%; 
                    height: auto; 
                    border-radius: 4px;
                }}
                .image-title {{ 
                    margin-top: 8px; 
                    font-size: 12px; 
                    color: #666;
                    word-break: break-all;
                }}
                .error {{ color: red; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h2>Related Images</h2>
            <div class="container">
        """
        
        for i, url in enumerate(image_urls):
            raw_url = convert_github_url_to_raw(url)
            filename = url.split('/')[-1]
            html_content += f"""
            <div class="image-card">
                <img src='{raw_url}' alt='{filename}' 
                     onerror="this.style.display='none'; this.nextSibling.style.display='block';">
                <div class="error" style="display: none;">
                    Failed to load: {filename}
                </div>
                <div class="image-title">{filename}</div>
            </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML file and open in browser
        html_file = "temp_images.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Open in default browser
        webbrowser.open(f'file://{os.path.abspath(html_file)}')
        print(f"📸 Opening {len(image_urls)} images in your browser...")
        
        return True
            
    except Exception as e:
        print(f"Error creating HTML display: {e}")
        return False

def collect_images_from_docs(docs):
    """Extract all unique image URLs from documents - FIXED VERSION"""
    all_images = []
    seen_images = set()
    
    for doc in docs:
        if hasattr(doc, 'metadata') and doc.metadata:
            # Get images string from metadata
            images_str = doc.metadata.get('images', '')
            
            # Split by delimiter to get individual image URLs
            if images_str and images_str.strip():
                # Split by pipe delimiter and filter out empty strings
                images = [img.strip() for img in images_str.split('|') if img.strip()]
                
                for img in images:
                    if img and img not in seen_images:
                        all_images.append(img)
                        seen_images.add(img)
    
    return all_images

def display_images_console(image_urls):
    """Display image information in console with clickable links"""
    if not image_urls:
        print("\n📸 No related images found.")
        return
    
    print(f"\n📸 Related Images ({len(image_urls)}):")
    print("-" * 50)
    
    for i, url in enumerate(image_urls, 1):
        # Convert to raw URL for direct viewing
        raw_url = convert_github_url_to_raw(url)
        filename = url.split('/')[-1]
        
        print(f"{i}. {filename}")
        print(f"   View: {raw_url}")
        print(f"   GitHub: {url}")
        print()

def ask_question(query, show_images=True, display_method='console'):
    try:
        # Get more relevant documents
        docs = vector_db.similarity_search(query, k=5)
        
        # Combine context more effectively
        context_parts = []
        for doc in docs:
            context_parts.append(doc.page_content)
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Collect images from documents
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
        print("\n🧠 Chatbot Answer:")
        print(response.text)
        
        # Display images if available
        if image_urls and show_images:
            if display_method == 'console':
                display_images_console(image_urls)
            elif display_method == 'matplotlib':
                print(f"\n📸 Displaying {len(image_urls)} related images...")
                for i, url in enumerate(image_urls):
                    print(f"\nImage {i+1}:")
                    display_image_from_url(url)
            elif display_method == 'html':
                print(f"\n📸 Creating HTML page with {len(image_urls)} related images...")
                if not display_images_html(image_urls):
                    # Fallback to console display
                    display_images_console(image_urls)
        
        # Debug: Show what context was used
        print(f"\n[Debug] Used {len(docs)} documents for context")
        if image_urls:
            print(f"[Debug] Found {len(image_urls)} related images")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your API key and internet connection.")

def debug_search(query):
    """Helper function to see what documents are being retrieved"""
    docs = vector_db.similarity_search(query, k=5)
    print(f"\n[Debug] Found {len(docs)} documents for query: '{query}'")
    
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        if hasattr(doc, 'metadata') and doc.metadata:
            print(f"Metadata: {doc.metadata}")
            
            # Show images if available - FIXED VERSION
            images_str = doc.metadata.get('images', '')
            if images_str:
                images = [img.strip() for img in images_str.split('|') if img.strip()]
                if images:
                    print(f"Associated Images ({len(images)}):")
                    for img in images:
                        print(f"  - {img}")

def interactive_image_viewer(image_urls):
    """Allow user to interactively view images"""
    if not image_urls:
        print("No images available.")
        return
    
    print(f"\nAvailable images ({len(image_urls)}):")
    for i, url in enumerate(image_urls, 1):
        filename = url.split('/')[-1]
        print(f"{i}. {filename}")
    
    while True:
        try:
            choice = input(f"\nEnter image number to view (1-{len(image_urls)}) or 'back': ")
            if choice.lower() == 'back':
                break
            
            idx = int(choice) - 1
            if 0 <= idx < len(image_urls):
                print(f"\nDisplaying: {image_urls[idx].split('/')[-1]}")
                display_image_from_url(image_urls[idx])
            else:
                print("Invalid number. Please try again.")
                
        except ValueError:
            print("Please enter a valid number or 'back'.")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    print("Enhanced Chatbot with Image Support initialized!")
    print("Commands:")
    print("  - Type your question normally")
    print("  - 'quit' to exit")
    print("  - 'debug: <question>' to see retrieved documents")
    print("  - 'images: <question>' to get interactive image viewer")
    print("  - 'settings' to change display preferences")
    print("  - 'open: <question>' to open images directly in browser")
    
    # Default settings
    show_images = True
    display_method = 'console'  # Options: 'console', 'matplotlib', 'html'
    
    while True:
        question = input("\nAsk a question: ")
        
        if question.lower() == 'quit':
            break
            
        elif question.lower() == 'settings':
            print("\nCurrent settings:")
            print(f"  Show Images: {show_images}")
            print(f"  Display Method: {display_method}")
            
            # Toggle image display
            toggle = input("\nShow images? (y/n): ").lower()
            show_images = toggle == 'y'
            
            if show_images:
                print("\nDisplay methods:")
                print("1. console - Show image links in console")
                print("2. matplotlib - Display images directly (requires GUI)")
                print("3. html - Create HTML page and open in browser")
                
                method_choice = input("Choose method (1-3): ")
                methods = {'1': 'console', '2': 'matplotlib', '3': 'html'}
                display_method = methods.get(method_choice, 'console')
            
            print(f"\nSettings updated: Images={show_images}, Method={display_method}")
            
        elif question.lower().startswith('debug:'):
            debug_query = question[6:].strip()
            debug_search(debug_query)
            
        elif question.lower().startswith('images:'):
            # Interactive image viewing mode
            img_query = question[7:].strip()
            docs = vector_db.similarity_search(img_query, k=5)
            image_urls = collect_images_from_docs(docs)
            
            print(f"\nQuery: {img_query}")
            interactive_image_viewer(image_urls)
            
        elif question.lower().startswith('open:'):
            # Quick browser opening mode
            open_query = question[5:].strip()
            docs = vector_db.similarity_search(open_query, k=5)
            image_urls = collect_images_from_docs(docs)
            
            if image_urls:
                print(f"\nOpening {len(image_urls)} images in browser for: {open_query}")
                display_images_html(image_urls)
            else:
                print(f"No images found for: {open_query}")
            
        else:
            ask_question(question, show_images, display_method)