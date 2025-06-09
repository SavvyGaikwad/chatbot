import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import shutil

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def flatten_json_to_docs(data):
    docs = []
    
    for section in data["content"]:
        title = section["title"]
        section_texts = []
        section_images = []
        
        print(f"Processing section: {title}")
        
        # Collect all content from this section (text and images)
        for item in section["content"]:
            if item["type"] in ["info", "step"]:
                text = item.get("text", "")
                if text:
                    section_texts.append(text)
                    print(f"  - Added text: {text[:50]}...")
            elif item["type"] == "media":
                image_path = item.get("path", "")
                if image_path:
                    section_images.append(image_path)
                    print(f"  - Added image: {image_path}")
        
        if section_texts or section_images:
            # Create one comprehensive document for the entire section
            full_content = f"Section: {title}\n\n" + "\n".join(section_texts)
            
            # Convert images list to a delimited string for ChromaDB compatibility
            images_str = "|".join(section_images) if section_images else ""
            
            # Add metadata with images as string
            metadata = {
                "section": title, 
                "type": "full_section",
                "images": images_str,  # Store as delimited string
                "has_images": len(section_images) > 0,
                "text_count": len(section_texts),
                "image_count": len(section_images)
            }
            
            docs.append(Document(
                page_content=full_content,
                metadata=metadata
            ))
            
            # Create individual documents for each step/info item
            for i, text in enumerate(section_texts):
                item_metadata = {
                    "section": title, 
                    "item": i, 
                    "type": "item",
                    "images": images_str,  # Store as delimited string
                    "has_images": len(section_images) > 0
                }
                
                docs.append(Document(
                    page_content=f"{title}: {text}",
                    metadata=item_metadata
                ))
    
    print(f"\nTotal documents created: {len(docs)}")
    return docs

def create_vector_db(json_path, persist_dir):
    # Remove existing database
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print(f"Removed existing database at {persist_dir}")
    
    data = load_json(json_path)
    docs = flatten_json_to_docs(data)
    
    # Show some sample documents with image info
    print("\nSample documents:")
    for i, doc in enumerate(docs[:3]):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")
        if doc.metadata.get('has_images'):
            images_list = doc.metadata.get('images', '').split('|') if doc.metadata.get('images') else []
            print(f"Associated Images: {images_list}")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    db.persist()
    print(f"\nSaved vector DB to {persist_dir}")
    
    # Test the database immediately
    print("\nTesting database...")
    test_queries = [
        "Forgot Password",
        "reset password",
        "How to Reset Your Password",
        "profile settings",
        "change password"
    ]
    
    for query in test_queries:
        results = db.similarity_search(query, k=3)
        print(f"\nQuery: '{query}' found {len(results)} results:")
        for j, result in enumerate(results):
            print(f"  Result {j+1}: {result.page_content[:100]}...")
            if result.metadata.get('has_images'):
                images_list = result.metadata.get('images', '').split('|') if result.metadata.get('images') else []
                print(f"    Associated Images: {images_list}")

def query_with_images(db_path, query, k=5):
    """
    Enhanced query function that returns both text and associated images
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    results = db.similarity_search(query, k=k)
    
    response_data = []
    
    for i, result in enumerate(results):
        # Convert images string back to list
        images_str = result.metadata.get("images", "")
        images_list = images_str.split('|') if images_str else []
        
        item = {
            "rank": i + 1,
            "content": result.page_content,
            "section": result.metadata.get("section", "Unknown"),
            "type": result.metadata.get("type", "Unknown"),
            "images": images_list,
            "has_images": result.metadata.get("has_images", False),
            "score": result.metadata.get("score", 0)  # If available
        }
        response_data.append(item)
    
    return response_data

def display_query_results(results):
    """
    Function to display query results in a formatted way
    """
    print(f"\nFound {len(results)} results:")
    print("=" * 80)
    
    for item in results:
        print(f"\nRank: {item['rank']}")
        print(f"Section: {item['section']}")
        print(f"Content: {item['content']}")
        
        if item['has_images']:
            print(f"Associated Images ({len(item['images'])}):")
            for img_path in item['images']:
                if img_path.strip():  # Only show non-empty image paths
                    print(f"  - {img_path}")
        else:
            print("No associated images")
        
        print("-" * 40)

# Example usage function
def example_usage():
    """
    Example of how to use the enhanced system
    """
    db_path = "vector_db"
    
    # Load the database and perform queries
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists(db_path):
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        # Test queries
        test_queries = [
            "How to reset forgotten password",
            "change profile settings", 
            "mobile device synchronization",
            "time zone preferences"
        ]
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print('='*60)
            
            results = query_with_images(db_path, query, k=3)
            display_query_results(results)
    else:
        print(f"Database not found at {db_path}. Please create it first.")

if __name__ == "__main__":
    # Create the vector database
    create_vector_db("data/User_Personal_Profile_Settings.json", "vector_db")
    
    # Run example usage
    print("\n" + "="*80)
    print("RUNNING EXAMPLE QUERIES WITH IMAGE SUPPORT")
    print("="*80)
    example_usage()