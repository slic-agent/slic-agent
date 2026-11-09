"""Pre-flight Ray resource checks before launching training or evaluation jobs."""

import ray
from typing import Dict, Optional, Tuple


def check_ray_resources(
    required_cpus: int,
    required_gpus: int = 0,
    env_workers: int = 0,
    env_cpus_per_worker: float = 1.0,
    verbose: bool = True,
    strict: bool = False,
) -> Tuple[bool, str]:
    """
    Check that the Ray cluster has enough GPUs and worker CPUs before launching a job.

    The CPU need is env_workers * env_cpus_per_worker (ray_init.num_cpus is only a cap).
    A GPU shortage or >= 2x CPU oversubscription fails; milder oversubscription only
    warns unless strict=True.

    Args:
        required_cpus: Base CPU requirement (for Ray init)
        required_gpus: Required GPU count
        env_workers: Number of environment workers
        env_cpus_per_worker: CPUs allocated per environment worker
        verbose: Print detailed resource information
        strict: If True, fail on CPU oversubscription. If False, just warn.

    Returns:
        Tuple of (success, message); message is the printable resource report.
    """
    if not ray.is_initialized():
        return False, "Ray is not initialized. Cannot check resources."
    
    try:
        # Get available resources
        available_resources = ray.available_resources()
        total_resources = ray.cluster_resources()
        
        # CPU need comes from the workers, not ray_init.num_cpus
        actual_cpu_needed = env_workers * env_cpus_per_worker
        
        # Get available CPUs (handle different keys)
        available_cpus = available_resources.get("CPU", 0)
        total_cpus = total_resources.get("CPU", 0)
        
        # Get available GPUs (handle GPU/NPU)
        available_gpus = available_resources.get("GPU", 0)
        if available_gpus == 0:
            available_gpus = available_resources.get("NPU", 0)
        
        total_gpus = total_resources.get("GPU", 0)
        if total_gpus == 0:
            total_gpus = total_resources.get("NPU", 0)
        
        # Build status message
        status_lines = []
        status_lines.append("=" * 70)
        status_lines.append("Ray Resource Check")
        status_lines.append("=" * 70)
        status_lines.append(f"Cluster Total Resources:")
        status_lines.append(f"  CPUs: {total_cpus:.1f}")
        status_lines.append(f"  GPUs: {total_gpus:.1f}")
        status_lines.append("")
        status_lines.append(f"Currently Available:")
        status_lines.append(f"  CPUs: {available_cpus:.1f}")
        status_lines.append(f"  GPUs: {available_gpus:.1f}")
        status_lines.append("")
        status_lines.append(f"Job Configuration:")
        status_lines.append(f"  ray_init.num_cpus: {required_cpus} (limit, not actual usage)")
        status_lines.append(f"  Environment workers: {env_workers}")
        status_lines.append(f"  CPUs per worker: {env_cpus_per_worker}")
        status_lines.append(f"  Actual CPU needed for workers: {actual_cpu_needed:.1f}")
        status_lines.append(f"  GPU requirement: {required_gpus}")
        status_lines.append("")
        
        # Check sufficiency
        cpu_sufficient = available_cpus >= actual_cpu_needed
        gpu_sufficient = available_gpus >= required_gpus
        
        # Calculate margins
        cpu_margin = available_cpus - actual_cpu_needed
        gpu_margin = available_gpus - required_gpus
        
        if cpu_sufficient and gpu_sufficient:
            status_lines.append("✓ Resource Check: PASSED")
            status_lines.append(f"  Available CPUs ({available_cpus:.1f}) >= Needed ({actual_cpu_needed:.1f})")
            status_lines.append(f"  CPU margin: +{cpu_margin:.1f} CPUs")
            status_lines.append(f"  GPU margin: +{gpu_margin:.1f} GPUs")
            success = True
        else:
            if not gpu_sufficient:
                # GPU shortage is always a hard failure
                status_lines.append("✗ Resource Check: FAILED - Insufficient GPUs")
                status_lines.append(f"  Available GPUs ({available_gpus:.1f}) < Needed ({required_gpus})")
                status_lines.append(f"  GPU deficit: {-gpu_margin:.1f} GPUs")
                status_lines.append("  Allocate more GPUs")
                success = False
            elif not cpu_sufficient:
                # CPU shortage - check severity
                cpu_utilization = actual_cpu_needed / available_cpus if available_cpus > 0 else float('inf')
                
                if cpu_utilization < 2.0:  # Less than 2x oversubscribed
                    status_lines.append("⚠ Resource Check: WARNING - Worker CPUs oversubscribed")
                    status_lines.append(f"  Available CPUs ({available_cpus:.1f}) < Needed ({actual_cpu_needed:.1f})")
                    status_lines.append(f"  CPU deficit: {-cpu_margin:.1f} CPUs")
                    status_lines.append(f"  Ray will schedule workers dynamically, but may queue some")
                    status_lines.append(f"  Consider reducing: batch_size or cpus_per_worker")
                    success = not strict  # Pass if not strict mode
                else:
                    status_lines.append("✗ Resource Check: FAILED - Severe worker CPU oversubscription")
                    status_lines.append(f"  Available CPUs ({available_cpus:.1f}) < Needed ({actual_cpu_needed:.1f})")
                    status_lines.append(f"  CPU deficit: {-cpu_margin:.1f} CPUs ({cpu_utilization:.1f}x oversubscribed)")
                    status_lines.append(f"  This will likely cause OOM or deadlocks")
                    status_lines.append(f"  Must reduce: batch_size or cpus_per_worker")
                    success = False
            else:
                success = True
        
        status_lines.append("=" * 70)
        
        message = "\n".join(status_lines)
        
        if verbose:
            print(message)
        
        return success, message
        
    except Exception as e:
        error_msg = f"Error checking Ray resources: {e}"
        if verbose:
            print(f"✗ {error_msg}")
        return False, error_msg


def suggest_resource_config(
    available_cpus: float,
    available_gpus: float,
    batch_size: int,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Suggest optimal resource configuration based on available resources.
    
    Args:
        available_cpus: Available CPU count
        available_gpus: Available GPU count
        batch_size: Desired batch size
        verbose: Print suggestions
        
    Returns:
        Dictionary with suggested configuration
    """
    # Reserve 1 CPU for system overhead
    usable_cpus = max(1, available_cpus - 1)
    
    # Allocate CPUs for Ray and environment workers
    # Rule of thumb: 70% for Ray base, 30% for workers
    ray_cpus = int(usable_cpus * 0.7)
    worker_cpus_total = usable_cpus - ray_cpus
    
    # Calculate workers
    num_workers = min(batch_size, int(worker_cpus_total / 0.5))  # Assuming 0.5 CPU per worker
    cpus_per_worker = worker_cpus_total / num_workers if num_workers > 0 else 0.5
    
    config = {
        "ray_init.num_cpus": ray_cpus,
        "data.val_batch_size": num_workers,
        "env.resources_per_worker.num_cpus": cpus_per_worker,
        "env.resources_per_worker.num_gpus": 0,
        "trainer.n_gpus_per_node": int(available_gpus),
    }
    
    if verbose:
        print("=" * 70)
        print("Suggested Resource Configuration")
        print("=" * 70)
        print(f"Available: {available_cpus} CPUs, {available_gpus} GPUs")
        print(f"Usable (after overhead): {usable_cpus} CPUs")
        print("")
        print("Suggested config:")
        for key, value in config.items():
            print(f"  {key}={value}")
        print("=" * 70)
    
    return config
