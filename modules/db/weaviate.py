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
  frame_index: int
  clip_id: int
  clip_name: str

class WeaviateRepository:
  def __init__(self):
    # self._WEAVIATE_URL = "http://localhost:8080"
    self._CLASS_NAME = "ClipFrame"
    self.__client = weaviate.connect_to_local()
    atexit.register(self.__client.close)

    self.__create_weaviate_collection()
  
  def __create_weaviate_collection(self) -> None:
    """
    Creates the data schema in Weaviate. This tells Weaviate what kind
    of data and properties to expect. It will not overwrite an existing schema.
    """
    if self.__client.collections.exists(self._CLASS_NAME):
      print(f"Collection '{self._CLASS_NAME}' already exists. Skipping creation.")
      return

    print(f"Creating collection '{self._CLASS_NAME}'...")
    
    self.__client.collections.create(
      name=self._CLASS_NAME,
      vectorizer_config=wvc.Configure.Vectorizer.none(), # no vectorizer
      properties=[
        wvc.Property(
          name="clip_id",
          data_type=wvc.DataType.INT,
          description="The identifier of the source video file.",
        ),
        wvc.Property(
          name="clip_name",
          data_type=wvc.DataType.TEXT,
          description="The name of the clip file"
        ),
        wvc.Property(
          name="frame_index",
          data_type=wvc.DataType.INT,
          description="The index of this frame within the clip.",
        )
      ]
    )
    print("Collection created successfully.")

  def upload_from_npy(self, clip_id: int, clip_name: str, data_path: str, batch_size: int = 100) -> None:
    print("Processing clip:", clip_name)

    # check existence
    if not os.path.exists(data_path):
      print(f"\tWarning: The directory {data_path} does not exist.")
      return

    data_properties = {
      "clip_id": clip_id,
      "clip_name": clip_name
    }

    frames = self.__client.collections.get(self._CLASS_NAME)

    with frames.batch.fixed_size(batch_size=batch_size) as batch:
      clip_path = os.path.join(data_path)
      try:
        vectors = np.load(clip_path)
      except Exception as e:
        print(f"\tError loading {clip_path}: {e}. Skipping.")
      
      for frame_id, vector in enumerate(tqdm(vectors, f"Upload clip {clip_name}")):        
        # Create a copy of the properties and add the specific frame_index
        current_properties = data_properties.copy()
        current_properties["frame_index"] = int(frame_id)
        
        # The new method is `add_object`, and the data is passed to `properties`
        batch.add_object(
          properties=current_properties,
          vector=vector.tolist()
        )
    
    print("Finished processing clip:", clip_name)
  
  def upload_from_folder(self, data_root_path: str, batch_size: int = 100) -> None:
    all_clips = [d for d in os.listdir(data_root_path)]

    for id, clip_name in enumerate(all_clips):
      video_data_path = os.path.join(data_root_path, clip_name)
      # print(video_data_path)
      self.upload_from_npy(data_path=video_data_path, clip_id=id, clip_name=clip_name, batch_size=100)

    print("\nAll videos have been processed.")

  def __format_query_results(self, results) -> list[QueryResult]:
    formatted_results = []
    for obj in results.objects:
      properties = obj.properties
      formatted_results.append(
        QueryResult(
          uuid=obj.uuid,
          frame_index=properties['frame_index'],
          clip_id=properties['clip_id'],
          clip_name=properties['clip_name']
        )
      )
    
    return formatted_results
  
  def query_by_vector(self, vector: list[int], k: int = 3) -> list[QueryResult]:
    my_collection = self.__client.collections.get(self._CLASS_NAME)
    response = my_collection.query.near_vector(
      near_vector=vector,
      limit=k
    )
    return self.__format_query_results(response)
  
  def query_by_text(self, text: str, k: int = 3) -> list[QueryResult]:
    my_collection = self.__client.collections.get(self._CLASS_NAME)
    response = my_collection.query.near_text(
      query=text,
      limit=k
    )
    return self.__format_query_results(response)
  
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