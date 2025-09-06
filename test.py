import os
import re
import time
from collections import Counter
from modules.db.weaviate_aidia import WeaviateRepository
from sentence_transformers import SentenceTransformer
from modules.utils.save_results import extract_query_info
from modules.utils.qa_system import QASystem

if __name__ == "__main__":
    # --- CONFIGURATION ---
    QUERIES_FOLDER = 'queries'
    # DATA_ROOT_BATCH1 = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/embeddings/embeddings_b1"
    DATA_ROOT_BATCH2 = "C:/Users/rokeM/Downloads/data/embeddings"
    VIDEO_FRAMES_ROOT_BATCH1 = "C:/Users/rokeM/Downloads/data/video_frames"
    SENTENCE_TRANSFORMER_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'
    VQA_MODEL_NAME = "Salesforce/blip2-opt-2.7b"
    number_of_results_per_query = 20

    weaviate_repo = WeaviateRepository()

    ############################################
    # --- DELETE DATA (If already exists) ---
    # print("\n--- Starting Data Deletion ---")
    # weaviate_repo.reset_database()
    # print("Weaviate database cleared.")

    # print("\n--- Starting Data Import ---")
    # weaviate_repo.import_embeddings(DATA_ROOT_BATCH1)
    # weaviate_repo.import_embeddings(DATA_ROOT_BATCH2)
    # print("--- Data Import Complete ---")

    # --- QUERY ---
    qa_system = QASystem()
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
    QUERY_TXT_FOLDER = os.path.join(os.path.dirname(__file__), QUERIES_FOLDER)
    query_files = [f for f in os.listdir(QUERY_TXT_FOLDER) if f.endswith('.txt')]

    print("Testing BLIP2 model...")
    test_image_path = "C:/Users/rokeM/Downloads/data/video_frames/Keyframes_L22/L22_V023/046.jpg"  # Use one you know exists
    qa_system.test_model(test_image_path)

    for filename in sorted(query_files):
        # Extract query ID and type from the filename
        query_id, query_type = extract_query_info(filename)
        
        if query_id is None or query_type is None:
            print(f"Cannot parse filename: {filename}")
            continue
        
        file_path = os.path.join(QUERY_TXT_FOLDER, filename)

        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as f:
            query_text = f.read().strip()

        print(f"\n--- Processing Query {query_id} (Type: {query_type.upper()}) ---")
        print("Query file:", filename)
        print("Query text:", query_text)

        if query_type == 'qa':
            qa_results = qa_system.answer_question(query_text, VIDEO_FRAMES_ROOT_BATCH1, number_of_results_per_query)
            print(f"QA Results: {qa_results}")
        else:
            print(f"Unsupported query type: {query_type}")
            
        print(f"Results saved for query {query_id} (type: {query_type})")
        print("-------")
    
    print("\n--- All queries processed successfully ---")