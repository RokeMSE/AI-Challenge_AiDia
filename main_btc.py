import os
import re
from modules.db.weaviate_btc import WeaviateRepository
from sentence_transformers import SentenceTransformer
from modules.utils.save_results_btc import extract_query_info, save_kis_results, save_qa_results, save_trake_results, create_btc_kis_result_object, create_btc_qa_result_object, create_btc_trake_result_object

if __name__ == "__main__":
    # --- CONFIGURATION ---
    QUERIES_FOLDER = 'queries'
    DATA_ROOT = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/embeddings_btc"
    SENTENCE_TRANSFORMER_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'
    number_of_results_per_query = 20

    weaviate_repo = WeaviateRepository()

    ############################################
    # --- DELETE DATA (If already exists) ---
    # print("\n--- Starting Data Deletion ---")
    # weaviate_repo.reset_database()
    # print("Weaviate database cleared.")

    # print("\n--- Starting Data Import ---")
    # weaviate_repo.import_embeddings(DATA_ROOT)
    # print("--- Data Import Complete ---")

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
                vector_index = r.vector_index
                distance = r.distance
            
                print("Video ID:", video_id)
                print("Frame name:", frame_name)
                print("Vector index:", vector_index)
                print("Distance:", distance)

                # Collect all results into a list using BTC helper
                kis_results_list.append(create_btc_kis_result_object(video_id, frame_name, vector_index))
            
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
                vector_index = r.vector_index
                distance = r.distance
            
                print("Video ID:", video_id)
                print("Frame name:", frame_name)
                print("Vector index:", vector_index)
                print("Distance:", distance)

                # Create and collect Q&A result using BTC helper
                answer_text = f"Answer for {query_text} (Placeholder)"
                qa_results_list.append(create_btc_qa_result_object(video_id, frame_name, vector_index, answer_text))
            
            # Save all collected results at once
            save_qa_results(qa_results_list, query_id=query_id)
            
        elif query_type == 'trake':
            res = weaviate_repo.query_by_vector(vector=embedding[0].tolist(), k=number_of_results_per_query)
            
            if not res:
                print("No results found for this query.")
                continue

            # Group results by video_id for TRAKE
            video_results = {}
            for r in res:
                video_id = r.video_id
                frame_name = r.frame_name
                vector_index = r.vector_index
                distance = r.distance
                
                if video_id not in video_results:
                    video_results[video_id] = []
                
                video_results[video_id].append((frame_name, vector_index, distance))

            print("Video results grouped:")
            for video_id, frame_data in video_results.items():
                print(f"Video ID: {video_id}")
                
                # Extract frame data for TRAKE result
                frame_tuples = [(frame_name, vector_index) for frame_name, vector_index, _ in frame_data]
                distances = [distance for _, _, distance in frame_data]
                
                print("Frame data:", frame_tuples)
                print("Distances:", distances)

                # Create and save TRAKE result for each video using BTC helper
                trake_result = create_btc_trake_result_object(video_id, frame_tuples)
                save_trake_results([trake_result], query_id=query_id)
        
        else:
            print(f"Unsupported query type: {query_type}")
            
        print(f"Results saved for query {query_id} (type: {query_type})")
        print("-------")
    
    print("\n--- All queries processed successfully ---")