import os
from modules.db.weaviate_aidia import WeaviateRepository
from modules.model.model import Clip4ClipHF
from sentence_transformers import SentenceTransformer
from modules.utils.get_queries import QueryLoader

if __name__ == "__main__":
    # --- CONFIGURATION ---
    QUERIES_FOLDER = 'queries'
    DATA_ROOT = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/embeddings"
    SENTENCE_TRANSFORMER_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'

    weaviate_repo = WeaviateRepository()
    # --- DELETE DATA (If already exists) ---
    # print("\n--- Starting Data Deletion ---")
    # weaviate_repo.reset_database()
    # print("Weaviate database cleared.")

    # --- IMPORT DATA ---
    # This will upload all your .npy files to the Weaviate database.
    # You only need to run this once, or when your embeddings change.
    # You can comment it out after the first successful run. Otherwise, RIP 
    # print("\n--- Starting Data Import ---")
    # weaviate_repo.import_embeddings(DATA_ROOT)
    # print("--- Data Import Complete ---")

    # --- QUERY ---
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
    QUERY_TXT_FOLDER = os.path.join(os.path.dirname(__file__), QUERIES_FOLDER)
    queries = QueryLoader(query_root_abs_path=QUERY_TXT_FOLDER).retrieve()

    embeddings = model.encode( [q.query_text for q in queries] )
    for id, emb in enumerate(embeddings):
    ### uncomment this if needed
        # q = queries[id]
        res = weaviate_repo.query_by_vector(vector=emb.tolist(), k=3)
        r = res[0]

        video_id = r.video_id
        frame_name = r.frame_name
        distance = r.distance
            
        print("Video ID:", video_id)
        print("Frame name:", frame_name)
        print("Distance:", distance)
        print("-------")