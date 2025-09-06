import os
import json
from modules.utils.qa_system import *
def test_detection_loading():
    print("=== Testing Object Detection Loading ===")
    
    # Update this path to your actual detection data folder
    detection_folder =  "C:/Users/rokeM/Downloads/data/objects/objects_b2" 
    
    loader = ObjectDetectionLoader(detection_folder)
    videos = [d for d in os.listdir(detection_folder) if os.path.isdir(os.path.join(detection_folder, d))]    
    video_id = videos[0]
    video_path = os.path.join(detection_folder, video_id)
    json_files = [f for f in os.listdir(video_path) if f.endswith('.json')]
    frame_name = os.path.splitext(json_files[0])[0]
    
    print(f"Testing with video: {video_id}, frame: {frame_name}")
    
    # Load frame analysis
    frame_analysis = loader.load_frame_analysis(video_id, frame_name)

    print(f"✅ Loaded frame analysis successfully")
    print(f"   Total objects: {len(frame_analysis.objects)}")
    print(f"   High confidence objects (>0.5): {len(frame_analysis.get_high_confidence_objects(0.5))}")
    print(f"   Entities: {list(frame_analysis.count_objects_by_entity().keys())[:10]}")
    
    return True

def test_answer_generation():
    print("\n=== Testing Answer Generation ===")

    detection_folder = "C:/Users/rokeM/Downloads/data/objects/objects_b2"
    loader = ObjectDetectionLoader(detection_folder)
    answer_generator = QAAnswerGenerator(loader)
    
    # Find a sample frame
    videos = [d for d in os.listdir(detection_folder) if os.path.isdir(os.path.join(detection_folder, d))]
    video_id = videos[0]
    video_path = os.path.join(detection_folder, video_id)
    json_files = [f for f in os.listdir(video_path) if f.endswith('.json')]
    frame_name = os.path.splitext(json_files[0])[0]
    
    frame_analysis = loader.load_frame_analysis(video_id, frame_name)
    
    test_questions = [
        "How many objects are in this image?",
        "What can you see in this frame?", 
        "Is there a tomato in this image?",
        "Where is the person located?",
        "Describe what you see in this frame"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = answer_generator.generate_answer(question, frame_analysis)
        print(f"A: {answer}")
    return True

def test_with_sample_data():
    print("\n=== Testing with Sample Data Format ===")
    
    # Create a sample frame analysis using the format from your 001.json
    sample_data = {
        "detection_scores": ["0.8645418", "0.641431", "0.6255185"],
        "detection_class_names": ["/m/07j87", "/m/07j87", "/m/0463sg"],
        "detection_class_entities": ["Tomato", "Tomato", "Fashion accessory"],
        "detection_boxes": [
            ["0.29378086", "0.47554952", "0.42751276", "0.5455454"],
            ["0.37215313", "0.5639252", "0.50209445", "0.6282254"],
            ["0.14457957", "0.32139552", "0.8178069", "0.72214806"]
        ],
        "detection_class_labels": ["392", "392", "277"]
    }
    
    objects = []
    for i, (score, class_name, entity, box, label) in enumerate(zip(
        sample_data['detection_scores'],
        sample_data['detection_class_names'], 
        sample_data['detection_class_entities'],
        sample_data['detection_boxes'],
        sample_data['detection_class_labels']
    )):
        objects.append(ObjectDetection(
            class_name=class_name,
            entity=entity,
            confidence=float(score),
            bbox=[float(coord) for coord in box],
            class_label=int(label)
        ))
    
    frame_analysis = FrameAnalysis("001", "TEST_VIDEO", objects)
    
    

    class MockLoader:
        pass
    answer_generator = QAAnswerGenerator(MockLoader())
    test_questions = [
        "How many tomatoes are in this image?",
        "Is there a tomato in this frame?",
        "What objects can you see?",
        "Where are the tomatoes located?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        answer = answer_generator.generate_answer(question, frame_analysis)
        print(f"A: {answer}")
    return True

if __name__ == "__main__":
    # Test with sample data (this will work without setting up paths)
    test_with_sample_data()
    
    # Uncomment these after updating the paths
    test_detection_loading()
    test_answer_generation()

