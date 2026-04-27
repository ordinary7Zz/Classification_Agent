"""
Main Classification Agent
Coordinates multiple models and uses the configured agent LLM to select the best prediction
"""

import yaml
import argparse
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np

from models import ModelRegistry
from agent import LLMClassificationAgent, AgentDecision
from utils import ImageProcessor


class ClassificationAgent:
    """
    Main classification agent that coordinates models and LLM-based decision-making
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the classification agent
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.image_processor = ImageProcessor(
            target_size=tuple(self.config['image_processing']['resize']),
            normalize_mean=tuple(self.config['image_processing']['normalize']['mean']),
            normalize_std=tuple(self.config['image_processing']['normalize']['std'])
        )
        
        self.model_registry = ModelRegistry()
        
        # Initialize LLM agent (see agent_llm in config)
        agent_llm_config = self.config['agent_llm']
        agent_config = self.config.get('agent', {})
        max_batch_size = agent_config.get('max_batch_size', 10)  # 从config读取，默认值为10
        top_k = max(1, int(agent_config.get('top_k', 1)))
        self.llm_agent = LLMClassificationAgent(
            api_key=agent_llm_config['api_key'],
            model_name=agent_llm_config['model_name'],
            temperature=agent_llm_config['temperature'],
            max_tokens=agent_llm_config['max_tokens'],
            max_batch_size=max_batch_size,
            top_k=top_k,
        )
        
        print(f"✓ Classification Agent initialized")
        print(f"✓ Agent LLM: {agent_llm_config['model_name']}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Returns:
            Configuration dictionary
        """
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def register_models_from_config(self) -> None:
        """
        Register models from configuration file
        Note: This is a placeholder - you need to implement actual model loading
        based on your specific models
        """
        models_config = self.config.get('models', {})
        
        print(f"\nNote: Found {len(models_config)} model configurations.")
        print("You need to implement specific model classes that inherit from BaseClassificationModel")
        print("and register them here. Example:")
        print("  from models.my_custom_model import MyCustomModel")
        print("  model = MyCustomModel(...)")
        print("  self.model_registry.register_model(model)")
    
    def classify(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        return_all_predictions: bool = False
    ) -> AgentDecision:
        """
        Run classification with all models and select best result
        
        Args:
            image_path: Path to input image
            mask_path: Optional path to segmentation mask
            return_all_predictions: Whether to return predictions from all models
            
        Returns:
            AgentDecision with selected best model
        """
        print(f"\n{'='*60}")
        print(f"Classification Agent - Processing Image")
        print(f"{'='*60}")
        
        # Load image and mask
        print(f"\n[1/4] Loading image: {image_path}")
        image = self.image_processor.load_image(image_path)
        print(f"      Image shape: {image.shape}")
        
        mask = None
        if mask_path:
            print(f"      Loading mask: {mask_path}")
            mask = self.image_processor.load_mask(mask_path)
            print(f"      Mask shape: {mask.shape}")
        
        # Run predictions with all models
        print(f"\n[2/4] Running inference with {len(self.model_registry)} models...")
        predictions = self.model_registry.predict_all(image, mask)
        
        if not predictions:
            raise RuntimeError("No predictions were generated")
        
        print(f"      Generated {len(predictions)} predictions")
        
        # Use agent LLM to select best model
        print(f"\n[3/4] Consulting agent LLM for best model selection...")
        decision = self.llm_agent.select_best_model(predictions)
        
        # Display results
        print(f"\n[4/4] Agent Decision:")
        print(f"{'='*60}")
        print(f"Selected Model: {decision.selected_model}")
        print(f"Predicted Class: {decision.selected_class}")
        print(f"Confidence: {decision.confidence:.4f}")
        print(f"\nReasoning:")
        print(f"  {decision.reasoning}")
        print(f"{'='*60}\n")
        
        return decision
    
    def classify_batch(
        self,
        image_paths: list,
        mask_paths: Optional[list] = None
    ) -> list:
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image paths
            mask_paths: Optional list of mask paths
            
        Returns:
            List of AgentDecisions
        """
        if mask_paths is None:
            mask_paths = [None] * len(image_paths)
        
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must match")
        
        decisions = []
        for img_path, mask_path in zip(image_paths, mask_paths):
            decision = self.classify(img_path, mask_path)
            decisions.append(decision)
        
        return decisions
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of the classification agent
        
        Returns:
            Status dictionary
        """
        return {
            'num_models': len(self.model_registry),
            'models': self.model_registry.list_models(),
            'agent_llm_model': self.config['agent_llm']['model_name'],
            'image_size': self.config['image_processing']['resize']
        }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Classification Agent - Multi-model classification with agent LLM selection"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--mask',
        type=str,
        default=None,
        help='Path to segmentation mask (optional)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = ClassificationAgent(config_path=args.config)
    
    # Register models (you need to implement this based on your models)
    agent.register_models_from_config()
    
    # Check if models are registered
    if len(agent.model_registry) == 0:
        print("\n" + "="*60)
        print("WARNING: No models registered!")
        print("="*60)
        print("\nTo use this agent, you need to:")
        print("1. Implement model classes that inherit from BaseClassificationModel")
        print("2. Register them in the model registry")
        print("\nExample:")
        print("  from models.base_model import BaseClassificationModel")
        print("  class MyModel(BaseClassificationModel):")
        print("      def load_model(self): ...")
        print("      def preprocess(self, image, mask): ...")
        print("      def predict(self, image, mask): ...")
        print("\n  agent.model_registry.register_model(MyModel(...))")
        print("="*60 + "\n")
        return
    
    # Run classification
    decision = agent.classify(
        image_path=args.image,
        mask_path=args.mask
    )
    
    # Export result
    print("\nFinal Result:")
    print(decision.to_dict())


if __name__ == "__main__":
    main()

