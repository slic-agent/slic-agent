"""
Detect and load FSDP sharded RL checkpoints for evaluation without converting them to HuggingFace format.
"""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from transformers import AutoConfig


def is_fsdp_checkpoint(checkpoint_path: str) -> bool:
    """
    Check if a directory contains FSDP sharded checkpoint files.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        True if FSDP checkpoint files are found, False otherwise
    """
    path = Path(checkpoint_path)
    if not path.exists() or not path.is_dir():
        return False
    
    # Look for FSDP model files
    fsdp_files = list(path.glob("model_world_size_*_rank_*.pt"))
    return len(fsdp_files) > 0


def detect_checkpoint_format(checkpoint_path: str) -> str:
    """
    Detect the format of a checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint directory or HuggingFace model ID (e.g. Qwen/Qwen2.5-1.5B-Instruct)
        
    Returns:
        One of: 'huggingface', 'fsdp', 'unknown'
    """
    path = Path(checkpoint_path)
    # Non-existent path: treat as HuggingFace Hub model ID (from_pretrained will download)
    if not path.exists():
        return 'huggingface'
    
    # Check for HuggingFace format (single file or sharded SafeTensors)
    has_hf_single = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    has_hf_sharded = (path / "model.safetensors.index.json").exists() or bool(
        list(path.glob("model-*-of-*.safetensors"))
    )
    has_hf_model = has_hf_single or has_hf_sharded
    
    # Check for FSDP format
    has_fsdp_model = is_fsdp_checkpoint(checkpoint_path)
    
    if has_hf_model:
        return 'huggingface'
    elif has_fsdp_model:
        return 'fsdp'
    else:
        return 'unknown'


def load_fsdp_checkpoint_state_dict(
    checkpoint_path: str,
    world_size: Optional[int] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load FSDP sharded checkpoint and merge into a single state dict.
    
    Args:
        checkpoint_path: Path to directory containing FSDP checkpoint files
        world_size: Number of ranks to load (auto-detected if None)
        device: Device to load tensors to
        
    Returns:
        Merged state dictionary
        
    Raises:
        ValueError: If checkpoint files are not found or invalid
    """
    path = Path(checkpoint_path)
    
    # Auto-detect world size if not provided
    if world_size is None:
        model_files = list(path.glob("model_world_size_*_rank_*.pt"))
        if not model_files:
            raise ValueError(f"No FSDP checkpoint files found in {checkpoint_path}")
        
        # Extract world size from first file
        filename = model_files[0].name
        # Format: model_world_size_{N}_rank_{R}.pt
        # Split: ["model", "world", "size", "N", "rank", "R.pt"]
        parts = filename.split("_")
        world_size = int(parts[3])  # Index 3 is the world size
        print(f"[FSDP Loader] Auto-detected world_size: {world_size}")
    
    print(f"[FSDP Loader] Loading FSDP checkpoint from: {checkpoint_path}")
    print(f"[FSDP Loader] World size: {world_size}")
    
    # Load and merge all shards
    merged_state_dict = {}
    
    for rank in range(world_size):
        model_file = path / f"model_world_size_{world_size}_rank_{rank}.pt"
        
        if not model_file.exists():
            raise ValueError(f"Missing shard file: {model_file}")
        
        print(f"[FSDP Loader] Loading shard {rank+1}/{world_size}: {model_file.name}")
        
        # Load shard
        shard_state_dict = torch.load(model_file, map_location=device, weights_only=False)
        
        # Merge into full state dict
        for key, value in shard_state_dict.items():
            if key in merged_state_dict:
                # This shouldn't happen with proper FSDP, but handle gracefully
                print(f"[FSDP Loader] Warning: Duplicate key {key} in shard {rank}, keeping first occurrence")
            else:
                merged_state_dict[key] = value
    
    print(f"[FSDP Loader] Loaded {len(merged_state_dict)} parameters")
    
    return merged_state_dict


def load_model_with_fsdp_support(
    model_class,
    checkpoint_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = False,
    device: str = "cpu",
    **kwargs
) -> Tuple[Any, str]:
    """
    Load a model from checkpoint with automatic FSDP support.
    
    This function automatically detects the checkpoint format and loads accordingly:
    - If HuggingFace format: Use standard from_pretrained()
    - If FSDP format: Load sharded state dict and initialize model
    
    Args:
        model_class: Transformers model class (e.g., AutoModelForCausalLM)
        checkpoint_path: Path to checkpoint directory
        torch_dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code
        device: Device to load model to
        **kwargs: Additional arguments for from_pretrained or model initialization
        
    Returns:
        Tuple of (model, checkpoint_format)
        
    Raises:
        ValueError: If checkpoint format cannot be determined or loaded
    """
    checkpoint_format = detect_checkpoint_format(checkpoint_path)
    
    # When path is a checkpoint root (e.g. .../global_step_150) with actor/ subdir, try actor/
    if checkpoint_format == 'unknown':
        path_obj = Path(checkpoint_path)
        if path_obj.is_dir():
            actor_path = path_obj / "actor"
            if actor_path.is_dir():
                actor_format = detect_checkpoint_format(str(actor_path))
                if actor_format != 'unknown':
                    checkpoint_path = str(actor_path)
                    checkpoint_format = actor_format
                    print(f"[Model Loader] Using actor subdirectory: {checkpoint_path}")
    
    print(f"[Model Loader] Checkpoint format detected: {checkpoint_format}")
    
    if checkpoint_format == 'huggingface':
        print(f"[Model Loader] Loading HuggingFace format checkpoint...")
        model = model_class.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        return model, checkpoint_format
        
    elif checkpoint_format == 'fsdp':
        print(f"[Model Loader] Loading FSDP format checkpoint...")
        
        # Use config from kwargs if provided (e.g. with attn_implementation), else load from checkpoint
        config = kwargs.pop('config', None)
        if config is None:
            config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=trust_remote_code)
        
        # Create model with meta tensors then load with assign=True to avoid "mixed torch.Tensor and DTensor"
        # when called from a process that has device_mesh (e.g. actor/ref init during PPO resume).
        from accelerate import init_empty_weights
        print(f"[Model Loader] Initializing model architecture from config (meta)...")
        with init_empty_weights():
            model = model_class.from_config(config, torch_dtype=torch_dtype, **kwargs)
        
        # Load FSDP sharded state dict (plain tensors on device)
        merged_state_dict = load_fsdp_checkpoint_state_dict(checkpoint_path, device=device)
        
        # Materialize from checkpoint; assign=True supports loading into meta tensors
        print(f"[Model Loader] Loading merged state dict into model (assign=True)...")
        load_result = model.load_state_dict(merged_state_dict, strict=False, assign=True)
        
        if load_result.missing_keys:
            print(f"[Model Loader] Warning: Missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys[:5]}...")
        if load_result.unexpected_keys:
            print(f"[Model Loader] Warning: Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:5]}...")
        
        print(f"[Model Loader] ✓ FSDP checkpoint loaded successfully")
        
        return model, checkpoint_format
        
    else:
        raise ValueError(
            f"Cannot determine checkpoint format for: {checkpoint_path}\n"
            f"Expected HuggingFace format (model.safetensors or pytorch_model.bin) "
            f"or FSDP format (model_world_size_*_rank_*.pt)"
        )


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        Dictionary with checkpoint information
    """
    path = Path(checkpoint_path)
    
    if not path.exists():
        return {"exists": False}
    
    info = {
        "exists": True,
        "path": str(path.absolute()),
        "format": detect_checkpoint_format(checkpoint_path),
    }
    
    # Check for common files
    info["has_config"] = (path / "config.json").exists()
    info["has_tokenizer"] = (path / "tokenizer.json").exists()
    info["has_hf_model"] = (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or (path / "model.safetensors.index.json").exists()
        or bool(list(path.glob("model-*-of-*.safetensors")))
    )
    info["has_fsdp_model"] = is_fsdp_checkpoint(checkpoint_path)
    
    # Count FSDP shards if applicable
    if info["has_fsdp_model"]:
        fsdp_files = list(path.glob("model_world_size_*_rank_*.pt"))
        info["fsdp_num_shards"] = len(fsdp_files)
        
        # Extract world size
        if fsdp_files:
            filename = fsdp_files[0].name
            # Format: model_world_size_{N}_rank_{R}.pt
            # Split: ["model", "world", "size", "N", "rank", "R.pt"]
            parts = filename.split("_")
            info["fsdp_world_size"] = int(parts[3])  # Index 3 is the world size
    
    return info
