# modules/db/weaviate.py
import weaviate
import weaviate.classes.config as wvc
import weaviate.classes as wc
import weaviate.classes.query as wvc_query
import numpy as np
import os
import atexit
import requests
import json
from tqdm import tqdm
from dataclasses import dataclass
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

@dataclass
class QueryResult:
  uuid: str
  video_id: str
  frame_name: str
  distance: float

class WeaviateRepository:
  _CLASS_NAME = "ClipFrame"

  def __init__(self):
    try:
      self.__client = weaviate.connect_to_local()
      # Ensure the client is closed when the program exits
      atexit.register(self.__client.close)
      self.__create_weaviate_collection()
    except Exception as e:
      print("ERROR: Could not connect to Weaviate.")
      print("Please ensure the Docker container is running with 'docker compose up -d'")
      raise e
  
  def __create_weaviate_collection(self) -> None:
    """
    Creates the data schema in Weaviate. This will not overwrite an existing collection.
    """
    if self.__client.collections.exists(self._CLASS_NAME):
      # print(f"Collection '{self._CLASS_NAME}' already exists. Skipping creation.")
      return

    print(f"Creating collection '{self._CLASS_NAME}'...")
    self.__client.collections.create(
      name=self._CLASS_NAME,
      vectorizer_config=wvc.Configure.Vectorizer.none(),
      properties=[
        wvc.Property(name="video_id", data_type=wvc.DataType.TEXT),
        wvc.Property(name="frame_name", data_type=wvc.DataType.TEXT),
      ]
    )
    print("Collection created successfully.")

  def import_embeddings(self, embeddings_folder: str, batch_size: int = 256):
    """
    Recursively finds all .npy files in the folder, and imports them into Weaviate.
    """
    collection = self.__client.collections.get(self._CLASS_NAME)
    
    print(f"Starting to import embeddings from: {embeddings_folder}")

    with collection.batch.dynamic() as batch:
      # Use os.walk to go through all subdirectories
      for root, _, files in tqdm(os.walk(embeddings_folder), desc="Scanning video folders"):
        for file in files:
          if file.endswith(".npy"):
            try:
              # Extract info from path
              video_id = os.path.basename(root)
              frame_name = os.path.splitext(file)[0]
              
              # Load the vector
              vector = np.load(os.path.join(root, file)).flatten().tolist()
              
              properties = {
                "video_id": video_id,
                "frame_name": frame_name,
              }
              
              batch.add_object(
                properties=properties,
                vector=vector
              )
            except Exception as e:
              print(f"Error processing file {file}: {e}")
    
    print("Finished importing all embeddings.")

  def __format_query_results(self, results) -> list[QueryResult]:
    """Formats the raw query response from Weaviate into a list of QueryResult objects."""
    return [
      QueryResult(
        uuid=obj.uuid,
        video_id=obj.properties['video_id'],
        frame_name=obj.properties['frame_name'],
        distance=obj.metadata.distance
      ) for obj in results.objects
    ]
  
  '''
  def query_by_vector(self, vector: list[float], k: int = 5) -> list[QueryResult]:
    """Queries Weaviate for the k nearest neighbors to the given vector."""
    collection = self.__client.collections.get(self._CLASS_NAME)
    response = collection.query.near_vector(
      near_vector=vector,
      limit=k,
      #return_metadata=wvc.query.MetadataQuery(distance=True) # Request distance metric
    )
    return self.__format_query_results(response)
  '''
  
  def query_by_vector(self, vector: list[float], k: int = 5) -> list[QueryResult]:
    """Queries Weaviate for the k nearest neighbors to the given vector."""
    try:
        # Get the collection (formerly class)
        collection = self.__client.collections.get(self._CLASS_NAME)

        # Perform the near vector query using the v4 client API
        response = collection.query.near_vector(
            near_vector=vector,
            limit=k,
            return_properties=["video_id", "frame_name"],
            return_metadata=wvc_query.MetadataQuery(distance=True) # Use wvc_query.MetadataQuery
        )

        results = []
        # Iterate through the objects in the response
        for obj in response.objects:
            uuid = str(obj.uuid) # Convert UUID object to string
            video_id = obj.properties.get("video_id")
            frame_name = obj.properties.get("frame_name")
            distance = obj.metadata.distance if obj.metadata else None # Access distance from metadata

            results.append(QueryResult(uuid=uuid, video_id=video_id, frame_name=frame_name, distance=distance))
        return results
    except Exception as e:
        print(f"Error querying vector: {e}")
        return []
    
  def reset_database(self):
    """
    Delete the entire ClipFrame collection and recreate an empty schema.
    """
    try:
      # Get the list of existing collections (as strings)
      existing_collections = self.__client.collections.list_all()

      # If ClipFrame exists, delete it
      if "ClipFrame" in existing_collections:
        self.__client.collections.delete("ClipFrame")
        print("Deleted collection ClipFrame")

      # Create a new empty schema
      self.__create_weaviate_collection()
      print("Recreated collection ClipFrame")

    except Exception as e:
      print(f"Error while resetting database: {e}")

  def query_with_paraphrases(self, query_text: str, model, k: int = 5, n_paraphrase: int = 5):
    """
    Generating paraphrases with Google AI Studio
    Query Weaviate for each vector and rerank
    """
    API_KEY = os.getenv("GOOGLE_API_KEY")
    MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

    # Call Google API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {
                "parts": [
                    {"text": f"Sinh {n_paraphrase} câu truy vấn tương đương với: '{query_text}' bằng tiếng anh."},
                    {"text": "Chỉ trả về câu thuần túy, mỗi câu 1 dòng."},
                    {"text": "Không format Markdown hay bất cứ ký tự đặc biệt nào ngoài chữ cái và dấu câu thông thường, không số thứ tự, không giải thích."}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    resp_json = response.json()

    #print(json.dumps(resp_json, indent=2, ensure_ascii=False))

    # Check for error
    if 'error' in resp_json:
        print(f"Google API returned an error: {resp_json['error']}")
        return [QueryResult(uuid="error", video_id="error", frame_name=query_text, distance=0.0)] 

    content_parts = resp_json['candidates'][0]['content']['parts']
    text = "\n".join([p['text'] for p in content_parts])

    # Get the paraphrases
    paraphrases = [line.strip("-• ") for line in text.split("\n") if line.strip()]
    paraphrases.append(query_text)

    vectors = [model.encode(p).tolist() for p in paraphrases]
    scores = defaultdict(float)
    seen = {}
    for v in vectors:
        results = self.query_by_vector(v, k=k)
        for r in results:
            scores[r.uuid] += 1 / (1 + r.distance)  
            seen[r.uuid] = r

    # Rerank
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [seen[uuid] for uuid, _ in ranked[:k]]
    