# modules/db/weaviate_btc.py
import weaviate
import weaviate.classes.config as wvc
import numpy as np
import os
import atexit
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class QueryResult:
  uuid: str
  video_id: str
  frame_name: str
  vector_index: int
  distance: float

class WeaviateRepository:
  _CLASS_NAME = "ClipFrameBTC"

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
        wvc.Property(name="vector_index", data_type=wvc.DataType.INT),
      ]
    )
    print("Collection created successfully.")

  def import_embeddings(self, embeddings_folder: str, batch_size: int = 256):
    """
    Recursively finds all .npy files in the folder, loads 2D arrays, and imports each vector into Weaviate.
    Each .npy file contains a 2D array where each row is a vector embedding.
    """
    collection = self.__client.collections.get(self._CLASS_NAME)
    
    print(f"Starting to import embeddings from: {embeddings_folder}")
    total_vectors_imported = 0

    with collection.batch.dynamic() as batch:
      # Use os.walk to go through all subdirectories
      for root, _, files in tqdm(os.walk(embeddings_folder), desc="Scanning folders"):
        for file in files:
          if file.endswith(".npy"):
            try:
              # Extract info from path
              video_id = os.path.basename(root)
              frame_name = os.path.splitext(file)[0]
              
              # Load the 2D array
              vectors_2d = np.load(os.path.join(root, file))
              
              # Check if it's actually a 2D array
              if vectors_2d.ndim != 2:
                print(f"Warning: {file} is not a 2D array (shape: {vectors_2d.shape}). Skipping.")
                continue
              
              # print(f"Processing {file}: shape {vectors_2d.shape} ({vectors_2d.shape[0]} vectors)")
              
              # Import each vector (row) separately
              for vector_index, vector in enumerate(vectors_2d):
                properties = {
                  "video_id": video_id,
                  "frame_name": frame_name,
                  "vector_index": vector_index,
                }
                
                batch.add_object(
                  properties=properties,
                  vector=vector.flatten().tolist()
                )
                total_vectors_imported += 1
                
            except Exception as e:
              print(f"Error processing file {file}: {e}")
    
    print(f"Finished importing all embeddings. Total vectors imported: {total_vectors_imported}")

  def __format_query_results(self, results) -> list[QueryResult]:
    """Formats the raw query response from Weaviate into a list of QueryResult objects."""
    return [
      QueryResult(
        uuid=obj.uuid,
        video_id=obj.properties['video_id'],
        frame_name=obj.properties['frame_name'],
        vector_index=obj.properties['vector_index'],
        distance=obj.metadata.distance
      ) for obj in results.objects
    ]
  
  def query_by_vector(self, vector: list[float], k: int = 5) -> list[QueryResult]:
    """Queries Weaviate for the k nearest neighbors to the given vector."""
    collection = self.__client.collections.get(self._CLASS_NAME)
    response = collection.query.near_vector(
      near_vector=vector,
      limit=k,
    #   return_metadata=wvc.query.MetadataQuery(distance=True) # Request distance metric
    )
    return self.__format_query_results(response)
  
  def reset_database(self):
    """
    Delete the entire ClipFrameBTC collection and recreate an empty schema.
    """
    try:
      # Get the list of existing collections (as strings)
      existing_collections = self.__client.collections.list_all()

      # If ClipFrameBTC exists, delete it
      if self._CLASS_NAME in existing_collections:
        self.__client.collections.delete(self._CLASS_NAME)
        print(f"Deleted collection {self._CLASS_NAME}")

      # Create a new empty schema
      self.__create_weaviate_collection()
      print(f"Recreated collection {self._CLASS_NAME}")

    except Exception as e:
      print(f"Error while resetting database: {e}")