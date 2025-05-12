"""
Utilities for optimizing large language models.
"""

import os
import gc
import torch
from typing import Optional, Dict, Any, Union

# Try to import optimization libraries
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    print("Warning: bitsandbytes not available. Quantization options will be limited.")

try:
    from peft import get_peft_model, LoraConfig, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Parameter-efficient fine-tuning options will be limited.")


class ModelOptimizer:
    """Helper class for optimizing large models for embedding generation."""
    
    @staticmethod
    def optimize_memory():
        """Free up CUDA memory by garbage collecting and emptying the cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_device_map(model_name: str) -> Union[str, Dict[str, Any]]:
        """
        Determine an appropriate device map strategy for a model.
        
        Args:
            model_name: Name or path of the model to load
            
        Returns:
            Device map configuration (auto, cpu, cuda, or detailed mapping)
        """
        if not torch.cuda.is_available():
            return "cpu"
            
        # Get GPU memory in GB
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Available GPU memory: {gpu_memory:.2f} GB")
        
        # If model has 'nomic' in the name, or is otherwise a large model
        if "nomic" in model_name.lower() or "7B" in model_name:
            if gpu_memory < 8:
                print("Warning: GPU memory may be insufficient for this model.")
                print("Using CPU fallback. This will be slow.")
                return "cpu"
            elif gpu_memory < 24:
                print("Using automatic device mapping to fit model in available memory")
                return "auto"
            else:
                print("Sufficient GPU memory available, loading model to CUDA")
                return "cuda:0"
        else:
            # For smaller models, we can usually fit them entirely on GPU
            return "cuda:0" if gpu_memory >= 4 else "cpu"
    
    @staticmethod
    def load_in_8bit(model_name: str) -> bool:
        """
        Determine if a model should be loaded in 8-bit precision.
        
        Args:
            model_name: Name or path of the model to load
            
        Returns:
            True if the model should be loaded in 8-bit precision
        """
        if not BNB_AVAILABLE:
            return False
            
        # Get GPU memory in GB
        if not torch.cuda.is_available():
            return False
            
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # If model has '7B' in the name, it's likely a 7 billion parameter model
        if "7B" in model_name and gpu_memory < 24:
            print("Using 8-bit precision to reduce memory footprint")
            return True
        
        return False

    @staticmethod
    def get_model_dtype(model_name: str) -> torch.dtype:
        """
        Determine the best dtype to use for a model based on hardware capabilities.
        
        Args:
            model_name: Name or path of the model to load
            
        Returns:
            Appropriate torch dtype
        """
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            print("Using bfloat16 precision")
            return torch.bfloat16
        elif torch.cuda.is_available():
            print("Using float16 precision")
            return torch.float16
        else:
            print("Using float32 precision (CPU)")
            return torch.float32
    
    @staticmethod
    def get_loading_options(model_name: str) -> Dict[str, Any]:
        """
        Get recommended loading options for a model.
        
        Args:
            model_name: Name or path of the model to load
            
        Returns:
            Dictionary of loading options
        """
        options = {
            "device_map": ModelOptimizer.get_device_map(model_name),
            "torch_dtype": ModelOptimizer.get_model_dtype(model_name),
        }
        
        # Add 8-bit loading if appropriate
        if ModelOptimizer.load_in_8bit(model_name):
            options["load_in_8bit"] = True
        
        return options