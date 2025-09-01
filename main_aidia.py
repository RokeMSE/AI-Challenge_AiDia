import os
import re
from modules.db.weaviate_aidia import WeaviateRepository
from sentence_transformers import SentenceTransformer
from modules.utils.save_results import extract_query_info, save_kis_results, save_qa_results, save_trake_results

# Tạo class động để lưu kết quả, phù hợp với định dạng của save_results.py
def create_kis_result_object(video_id, frame_index):
    return type('KISResult', (object,), {
        'video_id': video_id, 
        'frame_index': frame_index
    })()

def create_qa_result_object(video_id, frame_index, answer):
    return type('QAResult', (object,), {
        'video_id': video_id, 
        'frame_index': frame_index, 
        'answer': answer
    })()

def create_trake_result_object(video_id, frame_ids):
    return type('TrakeResult', (object,), {
        'video_id': video_id, 
        'frame_ids': frame_ids
    })()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    QUERIES_FOLDER = 'queries'
    DATA_ROOT = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/embeddings"
    SENTENCE_TRANSFORMER_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'
    number_of_results_per_query = 5

    weaviate_repo = WeaviateRepository()

    # --- QUERY ---
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
    QUERY_TXT_FOLDER = os.path.join(os.path.dirname(__file__), QUERIES_FOLDER)
    query_files = [f for f in os.listdir(QUERY_TXT_FOLDER) if f.endswith('.txt')]

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

        # Encode the current query
        embedding = model.encode([query_text])

        print(f"\n--- Processing Query {query_id} (Type: {query_type.upper()}) ---")
        print("Query file:", filename)
        print("Query text:", query_text)

        # Process each query type
        if query_type == 'kis':
            kis_results_list = []
            res = weaviate_repo.query_by_vector(vector=embedding[0].tolist(), k=number_of_results_per_query)
            
            if not res:
                print("No results found for this query.")
                continue

            for r in res:
                video_id = r.video_id
                frame_name = r.frame_name
                distance = r.distance
            
                print("Video ID:", video_id)
                print("Frame name:", frame_name)
                print("Distance:", distance)

                # Collect all results into a list
                kis_results_list.append(create_kis_result_object(video_id, frame_name))
            
            # Save all collected results at once
            save_kis_results(kis_results_list, query_id=query_id)
            
        elif query_type == 'qa':
            qa_results_list = []
            res = weaviate_repo.query_by_vector(vector=embedding[0].tolist(), k=number_of_results_per_query)
            
            if not res:
                print("No results found for this query.")
                continue

            for r in res:
                video_id = r.video_id
                frame_name = r.frame_name
                distance = r.distance
            
                print("Video ID:", video_id)
                print("Frame name:", frame_name)
                print("Distance:", distance)

                # Create and collect Q&A result (video_id, frame_index, "answer")
                answer_text = f"Answer for {query_text} (Placeholder)"
                qa_results_list.append(create_qa_result_object(video_id, frame_name, answer_text))
            
            # Save all collected results at once
            save_qa_results(qa_results_list, query_id=query_id)
            
        elif query_type == 'trake':
            res = weaviate_repo.query_by_vector(vector=embedding[0].tolist(), k=number_of_results_per_query)
            
            if not res:
                print("No results found for this query.")
                continue

            # Assuming all results are for the same video in TRAKE
            video_id = res[0].video_id
            frame_ids = [r.frame_name for r in res]
            distances = [r.distance for r in res]

            print("Video ID:", video_id)
            print("Frame IDs:", frame_ids)
            print("Distances:", distances)

            # Create and save a single TRAKE result object
            trake_result = create_trake_result_object(video_id, frame_ids)
            save_trake_results([trake_result], query_id=query_id)
        
        else:
            print(f"Unsupported query type: {query_type}")
            
        print(f"Results saved for query {query_id} (type: {query_type})")
        print("-------")
    
    print("\n--- All queries processed successfully ---")