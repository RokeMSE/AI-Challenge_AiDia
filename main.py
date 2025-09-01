from modules.db.weaviate import WeaviateRepository
from sentence_transformers import SentenceTransformer
from modules.utils.get_queries import QueryLoader
import os
############################################
### Configuration
# QUERY_FILE_PATH = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/queries.json" # Delete because we read from .txt
DATA_ROOT = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/embeddings"
SENTENCE_TRANSFORMER_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'
QUERIES_FOLDER = 'queries'
############################################

weaviate_repo = WeaviateRepository()

############################################
# --- DELETE DATA (If already exists) ---
# print("\n--- Starting Data Deletion ---")
# weaviate_repo.reset_database()
# print("Weaviate database cleared.")

### uncomment this to upload embeddings to weaviate
### NOTE: only run once to avoid duplicated data
# weaviate_repo.upload_from_folder(DATA_ROOT, batch_size=100)
############################################

model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)

QUERY_TXT_FOLDER = os.path.join(os.path.dirname(__file__), QUERIES_FOLDER)
queries = QueryLoader(query_root_abs_path=QUERY_TXT_FOLDER).retrieve()
# print(queries)

embeddings = model.encode( [q.query_text for q in queries] )
for id, emb in enumerate(embeddings):
  ### uncomment this if needed
  # q = queries[id]
  res = weaviate_repo.query_by_vector(vector=emb.tolist(), k=3)
  r = res[0]
  print("Clip name:", r.clip_name)
  print("Frame index:", r.frame_index)
  print("UUID:", r.uuid)
  print("-------")