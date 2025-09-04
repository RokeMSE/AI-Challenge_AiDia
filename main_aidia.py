import os
import re
import time
from collections import Counter
from modules.db.weaviate_aidia import WeaviateRepository
from sentence_transformers import SentenceTransformer
from modules.utils.save_results import extract_query_info, save_kis_results, save_qa_results, save_trake_results, create_kis_result_object, create_qa_result_object, create_trake_result_object
from modules.utils.search_utils import split_query_into_events

if __name__ == "__main__":
    # --- CONFIGURATION ---
    QUERIES_FOLDER = 'queries'
    DATA_ROOT = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/embeddings"
    SENTENCE_TRANSFORMER_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'
    number_of_results_per_query = 20

    weaviate_repo = WeaviateRepository()

    ############################################
    # --- DELETE DATA (If already exists) ---
    print("\n--- Starting Data Deletion ---")
    weaviate_repo.reset_database()
    print("Weaviate database cleared.")

    print("\n--- Starting Data Import ---")
    weaviate_repo.import_embeddings(DATA_ROOT)
    print("--- Data Import Complete ---")

    # --- QUERY ---
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
    
    # Download tokenizer and model EnViT5
    model_name = "VietAI/envit5-translation"
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

        print(f"\n--- Processing Query {query_id} (Type: {query_type.upper()}) ---")
        print("Query file:", filename)
        print("Query text:", query_text)

        # Process each query type
        if query_type == 'kis':
            print("KIS query type is currently not supported.")
            # # Encode the current query
            # embedding = model.encode([query_text])

            # kis_results_list = []
            # # res = weaviate_repo.query_by_vector(vector=embedding[0].tolist(), k=number_of_results_per_query)
            # res = weaviate_repo.query_with_paraphrases(query_text, model, k=10, n_paraphrase=10)
            
            # if not res:
            #     print("No results found for this query.")
            #     continue

            # for r in res:
            #     video_id = r.video_id
            #     frame_name = r.frame_name
            #     distance = r.distance
            
            #     print("Video ID:", video_id)
            #     print("Frame name:", frame_name)
            #     print("Distance:", distance)

            #     # Collect all results into a list
            #     kis_results_list.append(create_kis_result_object(video_id, frame_name))
            
            # # Save all collected results at once
            # save_kis_results(kis_results_list, query_id=query_id)
            
        elif query_type == 'qa':
            # Encode the current query
            embedding = model.encode([query_text])

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
            # Split query into sub-events
            sub = split_query_into_events(query_text)

            results_per_event = []
            for q in sub:
                # Search per event
                res = weaviate_repo.query_with_paraphrases(
                    query_text=q,
                    model=model,
                    k=100,          
                    n_paraphrase=7  
                )
                results_per_event.append(res)
                time.sleep(15)

            # Count video frequency
            video_counts = Counter()
            for res in results_per_event:
                video_counts.update([r.video_id for r in res if r])

            if not video_counts:
                print("No candidate videos found across events.")
                continue

            candidate_videos = [vid for vid, _ in video_counts.most_common(number_of_results_per_query)]

            all_trake_results = []

            for candidate in candidate_videos:
                candidate_frames = []
                candidate_score = 0.0

                for q in sub:
                    res = weaviate_repo.query_with_paraphrases(
                        query_text=q,
                        model=model,
                        k=50,
                        n_paraphrase=3
                    )
                    same_video = [r for r in res if r.video_id == candidate]
                    if same_video:
                        best = min(same_video, key=lambda r: r.distance)
                        candidate_frames.append(best.frame_name)
                        candidate_score += 1 / (1 + best.distance)

                if candidate_frames:
                    try:
                        candidate_frames = sorted(candidate_frames, key=lambda x: int(re.findall(r'\d+', x)[0]))
                    except:
                        candidate_frames = sorted(candidate_frames)

                    print("Candidate Video:", candidate, "Score:", candidate_score)
                    print("Frames:", candidate_frames)

                    trake_result = create_trake_result_object(candidate, candidate_frames)
                    all_trake_results.append(trake_result)

            if all_trake_results:
                save_trake_results(all_trake_results, query_id=query_id)
        else:
            print(f"Unsupported query type: {query_type}")
            
        print(f"Results saved for query {query_id} (type: {query_type})")
        print("-------")
    
    print("\n--- All queries processed successfully ---")