import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import re
from sentence_transformers import SentenceTransformer
from modules.db.weaviate_aidia import WeaviateRepository
from modules.utils.save_results import map_keyframes

@dataclass
class ObjectDetection:
    """Represents a detected object in a frame"""
    class_name: str
    entity: str
    confidence: float
    bbox: List[float]  # [y1, x1, y2, x2] normalized coordinates
    class_label: int

@dataclass
class FrameAnalysis:
    """Contains all object detections for a single frame"""
    frame_name: str
    video_id: str
    objects: List[ObjectDetection]
    
    def get_entities(self) -> List[str]:
        """Get all entity names in this frame"""
        return [obj.entity for obj in self.objects]
    
    def get_high_confidence_objects(self, threshold: float = 0.5) -> List[ObjectDetection]:
        """Get objects with confidence above threshold"""
        return [obj for obj in self.objects if obj.confidence >= threshold]
    
    def count_objects_by_entity(self) -> Dict[str, int]:
        """Count occurrences of each entity"""
        return Counter(self.get_entities())

class ObjectDetectionLoader:
    """Loads and processes object detection JSON files"""
    
    def __init__(self, detection_data_folder: str):
        self.detection_data_folder = detection_data_folder
    
    def load_frame_analysis(self, video_id: str, frame_name: str) -> Optional[FrameAnalysis]:
        """Load object detection data for a specific frame"""
        json_filename = f"{frame_name}.json"
        json_path = os.path.join(self.detection_data_folder, video_id, json_filename)
        
        if not os.path.exists(json_path):
            return None
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            objects = []
            scores = [float(s) for s in data['detection_scores']]
            class_names = data['detection_class_names']
            entities = data['detection_class_entities']
            boxes = [[float(coord) for coord in box] for box in data['detection_boxes']]
            labels = [int(l) for l in data['detection_class_labels']]            
            for i, (score, class_name, entity, box, label) in enumerate(
                zip(scores, class_names, entities, boxes, labels)
            ):
                objects.append(ObjectDetection(
                    class_name=class_name,
                    entity=entity,
                    confidence=score,
                    bbox=box,
                    class_label=label
                ))
            
            return FrameAnalysis(frame_name, video_id, objects)
        
        except Exception as e:
            print(f"Error loading detection data for {video_id}/{frame_name}: {e}")
            return None

class QAAnswerGenerator:
    """Generates answers for different types of questions"""
    def __init__(self, detection_loader: ObjectDetectionLoader):
        self.detection_loader = detection_loader
        
    def analyze_question_type(self, question: str) -> str:
        """Classify the type of question being asked"""
        question_lower = question.lower()
        
        # Count questions
        if any(word in question_lower for word in ['how many', 'count', 'số lượng', 'bao nhiêu']):
            return 'count'
        
        # Existence questions  
        if any(word in question_lower for word in ['is there', 'are there', 'có', 'exist']):
            return 'existence'
        
        # What questions
        if any(word in question_lower for word in ['what', 'gì', 'cái gì']):
            return 'what'
        
        # Where questions
        if any(word in question_lower for word in ['where', 'ở đâu', 'position']):
            return 'where'
        
        # Color questions
        if any(word in question_lower for word in ['color', 'màu', 'what color']):
            return 'color'
        
        # Description questions
        return 'description'
    
    def generate_answer(self, question: str, frame_analysis: FrameAnalysis) -> str:
        """Generate an answer based on the question and frame analysis"""
        if not frame_analysis or not frame_analysis.objects:
            return "I cannot see any objects clearly in this frame to answer your question."
        
        question_type = self.analyze_question_type(question)
        if question_type == 'count':
            return self._answer_count_question(question, frame_analysis)
        elif question_type == 'existence':
            return self._answer_existence_question(question, frame_analysis)
        elif question_type == 'what':
            return self._answer_what_question(question, frame_analysis)
        elif question_type == 'where':
            return self._answer_where_question(question, frame_analysis)
        else:
            return self._answer_description_question(question, frame_analysis)
    
    def _answer_count_question(self, question: str, frame_analysis: FrameAnalysis) -> str:
        """Answer counting questions"""
        # Extract objects mentioned in question
        entities_in_question = self._extract_entities_from_question(question)
        object_counts = frame_analysis.count_objects_by_entity()
        
        if entities_in_question:
            # Count specific objects mentioned in question
            total_count = 0
            found_entities = []
            for entity in entities_in_question:
                count = object_counts.get(entity, 0)
                if count > 0:
                    total_count += count
                    found_entities.append(f"{count} {entity}(s)")
            
            if found_entities:
                return f"I can see {', '.join(found_entities)} in this frame."
            else:
                return f"I don't see any {', '.join(entities_in_question)} in this frame."
        else:
            # Count all objects if no specific object mentioned
            high_conf_objects = frame_analysis.get_high_confidence_objects(0.3)
            return f"I can see {len(high_conf_objects)} objects in total in this frame."
    
    def _answer_existence_question(self, question: str, frame_analysis: FrameAnalysis) -> str:
        """Answer yes/no existence questions"""
        entities_in_question = self._extract_entities_from_question(question)
        entities_in_frame = set(frame_analysis.get_entities())
        if entities_in_question:
            found = []
            not_found = []
            
            for entity in entities_in_question:
                if entity in entities_in_frame:
                    found.append(entity)
                else:
                    not_found.append(entity)
            
            if found and not not_found:
                return f"Yes, I can see {', '.join(found)} in this frame."
            elif found and not_found:
                return f"I can see {', '.join(found)} but not {', '.join(not_found)} in this frame."
            else:
                return f"No, I don't see any {', '.join(entities_in_question)} in this frame."
        else:
            return "Please specify what object you're asking about."
    
    def _answer_what_question(self, question: str, frame_analysis: FrameAnalysis) -> str:
        """Answer 'what' questions"""
        high_conf_objects = frame_analysis.get_high_confidence_objects(0.3)
        if not high_conf_objects:
            return "I cannot clearly identify any specific objects in this frame."
        
        # Get most confident objects
        sorted_objects = sorted(high_conf_objects, key=lambda x: x.confidence, reverse=True)
        top_objects = sorted_objects[:5]  # Top 5 most confident
        
        entities = [obj.entity for obj in top_objects]
        entity_counts = Counter(entities)
        
        descriptions = []
        for entity, count in entity_counts.most_common():
            if count == 1:
                descriptions.append(entity)
            else:
                descriptions.append(f"{count} {entity}s")
        
        return f"I can see: {', '.join(descriptions)}."
    
    def _answer_where_question(self, question: str, frame_analysis: FrameAnalysis) -> str:
        """Answer location questions"""
        entities_in_question = self._extract_entities_from_question(question)
        
        if not entities_in_question:
            return "Please specify what object you're asking about the location of."
        
        found_objects = []
        for obj in frame_analysis.objects:
            if obj.entity in entities_in_question and obj.confidence > 0.3:
                position = self._describe_position(obj.bbox)
                found_objects.append(f"{obj.entity} is located {position}")
        
        if found_objects:
            return ". ".join(found_objects) + "."
        else:
            return f"I cannot locate {', '.join(entities_in_question)} in this frame."
    
    def _answer_description_question(self, question: str, frame_analysis: FrameAnalysis) -> str:
        """Answer general description questions"""
        high_conf_objects = frame_analysis.get_high_confidence_objects(0.2)
        
        if not high_conf_objects:
            return "I cannot clearly see the details in this frame."
        
        entity_counts = Counter([obj.entity for obj in high_conf_objects])
        descriptions = []
        
        for entity, count in entity_counts.most_common(10):  # Top 10
            if count == 1:
                descriptions.append(entity)
            else:
                descriptions.append(f"{count} {entity}s")
        
        return f"In this frame, I can see: {', '.join(descriptions)}."
    
    def _extract_entities_from_question(self, question: str) -> List[str]:
        """Extract potential object entities from the question"""
        common_objects = [
            'person', 'people', 'man', 'woman', 'child', 'baby',
            'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle',
            'dog', 'cat', 'bird', 'animal',
            'table', 'chair', 'sofa', 'bed',
            'phone', 'computer', 'tv', 'laptop',
            'book', 'bottle', 'cup', 'bowl',
            'apple', 'banana', 'orange', 'fruit',
            'tree', 'flower', 'plant',
            'ball', 'toy', 'game',
            'tomato', 'carrot', 'vegetable' 
        ] # Why the heck is it all English ? Thought this was a Vietnamese contest :)
        
        question_lower = question.lower()
        found_entities = []
        
        for entity in common_objects:
            if entity in question_lower:
                found_entities.append(entity.title())
        
        return found_entities
    
    def _describe_position(self, bbox: List[float]) -> str:
        """Describe the position of an object based on its bounding box"""
        y1, x1, y2, x2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Horizontal position
        if center_x < 0.33:
            h_pos = "on the left"
        elif center_x > 0.67:
            h_pos = "on the right"
        else:
            h_pos = "in the center"
        
        # Vertical position  
        if center_y < 0.33:
            v_pos = "at the top"
        elif center_y > 0.67:
            v_pos = "at the bottom"
        else:
            v_pos = "in the middle"
        
        return f"{h_pos} {v_pos}"

class QASystem:
    """QA system that combines visual similarity and object detection"""
    def __init__(self, weaviate_repo: WeaviateRepository, detection_data_folder: str):
        self.weaviate_repo = weaviate_repo
        self.detection_loader = ObjectDetectionLoader(detection_data_folder)
        self.answer_generator = QAAnswerGenerator(self.detection_loader)
        self.sentence_model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
    
    def answer_question(self, question: str, k: int = 20) -> List[Tuple[str, str, str]]:
        """
        Answer a question by finding relevant frames and generating answers
        Returns list of (video_id, frame_index, answer) tuples
        """
        # 1. Find visually relevant frames using CLIP embeddings
        embedding = self.sentence_model.encode([question])
        search_results = self.weaviate_repo.query_by_vector(
            vector=embedding[0].tolist(), 
            k=k
        )
        
        qa_results = []
        
        for result in search_results:
            video_id = result.video_id
            frame_name = result.frame_name
            
            # 2. Load object detection data for this frame
            frame_analysis = self.detection_loader.load_frame_analysis(video_id, frame_name)
            
            # 3. Generate answer based on the question and detected objects
            if frame_analysis:
                answer = self.answer_generator.generate_answer(question, frame_analysis)
            else:
                answer = "Unable to analyze objects in this frame."
            
            # 4. Map frame name to frame index (using your existing mapping function)
            frame_index = map_keyframes(video_id, frame_name)
            
            qa_results.append((video_id, frame_index, answer))
        
        return qa_results
    
    def get_detailed_frame_info(self, video_id: str, frame_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about objects in a specific frame"""
        frame_analysis = self.detection_loader.load_frame_analysis(video_id, frame_name)
        
        if not frame_analysis:
            return None
        
        return {
            'video_id': video_id,
            'frame_name': frame_name,
            'total_objects': len(frame_analysis.objects),
            'high_confidence_objects': len(frame_analysis.get_high_confidence_objects(0.5)),
            'entity_counts': frame_analysis.count_objects_by_entity(),
            'top_objects': [
                {
                    'entity': obj.entity,
                    'confidence': obj.confidence,
                    'position': self.answer_generator._describe_position(obj.bbox)
                }
                for obj in sorted(frame_analysis.objects, key=lambda x: x.confidence, reverse=True)[:10]
            ]
        }