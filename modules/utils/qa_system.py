import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from modules.db.weaviate_aidia import WeaviateRepository
from sentence_transformers import SentenceTransformer
import os
from PIL import Image

class QASystem:
    """
    There are two main components in this QA system:
    1. Retrieval: Use Weaviate and SentenceTransformer to find relevant frames.
    2. Reader: Use VQA model (BLIP2) to read and answer questions from candidate frames.
    """
    def __init__(self, sentence_transformer_model='clip-ViT-B-32-multilingual-v1', vqa_model_name="Salesforce/blip2-flan-t5-xl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # --- Retrieval ---
        self.retrieval_model = SentenceTransformer(sentence_transformer_model, device=self.device)
        self.weaviate_repo = WeaviateRepository()

        # --- Reader (VQA) - BLIP2 Configuration ---
        print(f"Loading BLIP2 model: {vqa_model_name}")
        self.vqa_processor = Blip2Processor.from_pretrained(vqa_model_name)
        self.vqa_model = Blip2ForConditionalGeneration.from_pretrained(
            vqa_model_name,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,  # Use float16 for GPU efficiency
            device_map="auto" if self.device == "cuda" else None
        )
        
        # If not using device_map="auto", manually move to device
        if self.device == "cuda" and "device_map" not in locals():
            self.vqa_model.to(self.device)

    def test_model(self, image_path: str):
        """
        Simple test function to check if BLIP2 is working at all
        """
        try:
            print(f"Testing BLIP2 model with image: {image_path}")
            raw_image = Image.open(image_path).convert('RGB')
            print(f"Image loaded: {raw_image.size}")
            
            # Test basic image description
            inputs = self.vqa_processor(images=raw_image, text="What do you see?", return_tensors="pt").to(self.device)
            print("Inputs prepared")
            
            with torch.no_grad():
                outputs = self.vqa_model.generate(
                    **inputs, 
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.vqa_processor.tokenizer.eos_token_id
                )
            
            result = self.vqa_processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"Model output: '{result}'")
            return result
            
        except Exception as e:
            print(f"Error in test: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def answer_question(self, query_text: str, video_frames_root: str, k_retrieval: int = 5):
        """
        Returns:
            list: list of candidates (video_id, frame_index, answer).
        """
        print(f"Received query: '{query_text}'")

        # Retrieving top k candidate frames
        query_embedding = self.retrieval_model.encode([query_text])
        candidate_frames = self.weaviate_repo.query_by_vector(vector=query_embedding[0].tolist(), k=k_retrieval)

        if not candidate_frames:
            print("No relevant frames found.")
            return []
        
        print(f"Found {len(candidate_frames)} candidates. Best candidate: Video {candidate_frames[0].video_id}, Frame {candidate_frames[0].frame_name} (Distance: {candidate_frames[0].distance:.4f})")

        # Analyzing candidate frames with BLIP2 model to find the answer
        final_answers = []
        for candidate in candidate_frames:
            try:
                l_prefix = candidate.video_id.split('_')[0]  # Get the folder prefix, ex "L26"
                intermediate_folder = f"Keyframes_{l_prefix}"  # create folder name
                frame_image_path = os.path.join(video_frames_root, intermediate_folder, candidate.video_id, f"{candidate.frame_name}.jpg")
            except IndexError:
                print(f"Warning: Could not parse video_id '{candidate.video_id}' for intermediate path. Using direct path.")
                frame_image_path = os.path.join(video_frames_root, candidate.video_id, f"{candidate.frame_name}.jpg")
            
            if not os.path.exists(frame_image_path):
                print(f"Frame image not found at {frame_image_path}. Trying .png")
                frame_image_path = frame_image_path.replace('.jpg', '.png')
                if not os.path.exists(frame_image_path):
                    print(f"Frame image not found at {frame_image_path} either. Skipping.")
                    continue

            raw_image = Image.open(frame_image_path).convert('RGB')

            # BLIP2 VQA - Question Answering
            try:
                inputs = self.vqa_processor(images=raw_image, text=query_text, return_tensors="pt").to(self.device)
                
                # Generate answer with appropriate parameters for BLIP2
                with torch.no_grad():
                    generated_ids = self.vqa_model.generate(
                        **inputs,
                        max_new_tokens=50,  # Reduced for more concise answers
                        min_length=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.vqa_processor.tokenizer.eos_token_id
                    )
                
                # Decode the answer
                answer = self.vqa_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                # Clean up the answer and filter out irrelevant responses
                if query_text.lower() in answer.lower():
                    # Try to extract just the answer part
                    answer_parts = answer.split(query_text)
                    if len(answer_parts) > 1:
                        answer = answer_parts[-1].strip()
                
                # Filter out URLs and irrelevant answers
                if any(url_indicator in answer.lower() for url_indicator in ['http', 'www.', '.com', 'youtu.be', 'youtube']):
                    print(f"  - Analyzing {candidate.video_id}/{candidate.frame_name}: Skipping URL answer -> '{answer}'")
                    continue
                
                print(f"  - Analyzing {candidate.video_id}/{candidate.frame_name}: Answer found -> '{answer}'")
                
                if answer and len(answer.strip()) > 0:
                    final_answers.append({
                        "video_id": candidate.video_id,
                        "frame_name": candidate.frame_name,
                        "answer": answer,
                        "distance": candidate.distance  # Include retrieval confidence
                    })
                    break  # Stop at the first valid answer
                    
            except Exception as e:
                print(f"Error processing frame {candidate.video_id}/{candidate.frame_name}: {str(e)}")
                continue
        
        return final_answers

    def answer_question_multiple_candidates(self, query_text: str, video_frames_root: str, k_retrieval: int = 5, max_answers: int = 3):
        """
        Alternative method that returns multiple answers from different candidates.
        
        Returns:
            list: list of candidates with answers, ranked by retrieval confidence.
        """
        print(f"Received query: '{query_text}' (multiple candidates mode)")

        # Retrieving top k candidate frames
        query_embedding = self.retrieval_model.encode([query_text])
        candidate_frames = self.weaviate_repo.query_by_vector(vector=query_embedding[0].tolist(), k=k_retrieval)

        if not candidate_frames:
            print("No relevant frames found.")
            return []
        
        print(f"Found {len(candidate_frames)} candidates.")

        # Analyzing multiple candidate frames
        final_answers = []
        for i, candidate in enumerate(candidate_frames[:max_answers]):
            try:
                l_prefix = candidate.video_id.split('_')[0]
                intermediate_folder = f"Keyframes_{l_prefix}"
                frame_image_path = os.path.join(video_frames_root, intermediate_folder, candidate.video_id, f"{candidate.frame_name}.jpg")
            except IndexError:
                frame_image_path = os.path.join(video_frames_root, candidate.video_id, f"{candidate.frame_name}.jpg")
            
            if not os.path.exists(frame_image_path):
                frame_image_path = frame_image_path.replace('.jpg', '.png')
                if not os.path.exists(frame_image_path):
                    continue

            raw_image = Image.open(frame_image_path).convert('RGB')

            try:
                inputs = self.vqa_processor(images=raw_image, text=query_text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.vqa_model.generate(
                        **inputs,
                        max_new_tokens=50,
                        min_length=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.vqa_processor.tokenizer.eos_token_id
                    )
                
                answer = self.vqa_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                # Clean up answer if needed
                if query_text.lower() in answer.lower():
                    answer_parts = answer.split(query_text)
                    if len(answer_parts) > 1:
                        answer = answer_parts[-1].strip()
                
                if answer and len(answer.strip()) > 0:
                    final_answers.append({
                        "video_id": candidate.video_id,
                        "frame_name": candidate.frame_name,
                        "answer": answer,
                        "distance": candidate.distance,
                        "rank": i + 1
                    })
                    print(f"  - Candidate {i+1}: {candidate.video_id}/{candidate.frame_name} -> '{answer}'")
                    
            except Exception as e:
                print(f"Error processing candidate {i+1}: {str(e)}")
                continue
        
        return final_answers
