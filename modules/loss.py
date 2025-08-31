import numpy as np
from abc import ABC, abstractmethod
class LossCalculator(ABC):
  @abstractmethod
  def calc(self, predictions, targets):
    pass

class CosineSimilarityLoss(LossCalculator):
  def __init__(self):
    super().__init__()

  def calc(self, predictions, targets):
    # The vector shape must be (1,D)
    if predictions.shape[0] != 1 or targets.shape[0] != 1:
      raise ValueError("Input vectors must have shape (1, D)")
    
    # The vector dimension must be the same
    if predictions.shape[1] != targets.shape[1]:
      raise ValueError("Input vectors must have the same dimension")

    # Adjust the vector to avoid division by zero
    epsilon = 1e-8
    predictions = predictions + epsilon
    targets = targets + epsilon
    
    # Normalize the vectors to unit length
    norm_predictions = predictions / np.linalg.norm(predictions, axis=1, keepdims=True)
    norm_targets = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    
    # Compute cosine similarity
    cosine_sim = np.sum(norm_predictions * norm_targets, axis=1)
    
    # Cosine similarity loss
    loss = 1 - cosine_sim
    return loss