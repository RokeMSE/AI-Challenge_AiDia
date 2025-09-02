import os
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass
class Query:
  query_name: str
  query_text: str

class QueryLoader:
  def __init__(self, query_root_abs_path: str):
    self.query_root_abs_path = query_root_abs_path

  def retrieve(self) -> list[Query]:
    queries = []
    if not os.path.isdir(self.query_root_abs_path):
      raise ValueError(f"{self.query_root_abs_path} is not a valid directory")
    
    # Sort filenames to ensure consistent processing order
    for filename in sorted(os.listdir(self.query_root_abs_path)):
      if filename.endswith('.txt'):
        file_path = os.path.join(self.query_root_abs_path, filename)
        try:
          with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
              queries.append(Query(query_name=filename, query_text=content))
        except Exception as e:
          print(f"Error reading file {filename}: {e}")
          continue
    
    print(f"Loaded {len(queries)} queries from {self.query_root_abs_path}")
    return queries