from modules.db.weaviate import WeaviateRepository
from sentence_transformers import SentenceTransformer
from modules.utils.get_queries import QueryLoader
############################################
### Configuration
QUERY_FILE_PATH = "/home/ketamean/Documents/Y3/AIC/data/queries/query-p1-groupA"
DATA_ROOT = "/home/ketamean/Documents/Y3/AIC/data/clip-features-32-aic25-b1/clip-features-32"
SENTENCE_TRANSFORMER_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'
############################################


weaviate_repo = WeaviateRepository()
############################################
### uncomment this to upload embeddings to weaviate
### NOTE: only run once to avoid duplicated data
# weaviate_repo.upload_from_folder(DATA_ROOT, batch_size=100)
############################################
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL_NAME)
queries = QueryLoader(query_root_abs_path=QUERY_FILE_PATH).retrieve()
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