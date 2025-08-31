from modules.db.weaviate import WeaviateRepository
from sentence_transformers import SentenceTransformer
from modules.utils.get_queries import QueryLoader
import os
############################################
### Configuration
QUERY_FILE_PATH = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/queries.json"
DATA_ROOT = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/embeddings"
SENTENCE_TRANSFORMER_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'
QUERIES_FOLDER = 'queries'
############################################


weaviate_repo = WeaviateRepository()
############################################
### uncomment this to upload embeddings to weaviate
### NOTE: only run once to avoid duplicated data
# weaviate_repo.upload_from_folder(DATA_ROOT, batch_size=100)

# Delete the whole database after uploading
# Only run when you want to delete all data in Weaviate
# weaviate_repo.delete_all()  # Uncomment to use
############################################

model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)

# Automatically read all .txt files in the 'queries' folder and create a list of queries
QUERY_TXT_FOLDER = os.path.join(os.path.dirname(__file__), QUERIES_FOLDER)
queries = []
for filename in os.listdir(QUERY_TXT_FOLDER):
  if filename.endswith('.txt'):
    file_path = os.path.join(QUERY_TXT_FOLDER, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
      content = f.read().strip()
      if content:
        queries.append(content)

print(f"Number of queries: {len(queries)}")
for idx, q in enumerate(queries):
  print(f"Query {idx+1}: {q}")

embeddings = model.encode(queries)
for id, emb in enumerate(embeddings):
  res = weaviate_repo.query_by_vector(vector=emb.tolist(), k=3)
  r = res[0]
  print("Clip name:", r.clip_name)
  print("Frame index:", r.frame_index)
  print("UUID:", r.uuid)
  print("-------")