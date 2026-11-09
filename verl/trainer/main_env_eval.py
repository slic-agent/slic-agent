#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate pretrained model on interactive environments WITHOUT training.
Runs rollouts using the same environment and trajectory collection as PPO,
but skips all gradient updates.
"""

import os
import json
import time
from math import comb
import hydra
import ray
import numpy as np
from omegaconf import OmegaConf
from verl.utils.episode_metrics import episode_boundaries_from_lengths

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@hydra.main(config_path="config", config_name="env_evaluation", version_base=None)
def main(config):
    run_env_evaluation(config)


def _normalize_rollout_scalar(key, value):
    """Preserve raw measurement precision while converting NumPy scalars."""
    import numpy as np
    return value.item() if isinstance(value, np.generic) else value


def _normalize_rollout_entry(entry):
    return {k: _normalize_rollout_scalar(k, v) for k, v in entry.items()}


def _step_info_to_metrics(step_info):
    if not isinstance(step_info, dict):
        return {}
    return {
        "gain_dB": step_info.get("gain_dB"),
        "differential_gain_db": step_info.get("differential_gain_db"),
        "power_uW": step_info.get("power_uW"),
        "bandwidth_MHz": step_info.get("bandwidth_MHz"),
        # without these, the in-band gate / strict-missing rules cannot
        # fire inside check_core_success at the eval summary layer.
        "bandwidth_kind": step_info.get("bandwidth_kind"),
        "inband_flatness_db": step_info.get("inband_flatness_db"),
        "phase_margin_status": step_info.get("phase_margin_status"),
        "phase_margin_deg": step_info.get("phase_margin_deg"),
        "offset_v": step_info.get("offset_v"),
        "cmrr_db": step_info.get("cmrr_db"),
        "psrr_plus_db": step_info.get("psrr_plus_db"),
        "psrr_minus_db": step_info.get("psrr_minus_db"),
    }


def _compute_core_success_from_step_info(step_info):
    if not isinstance(step_info, dict):
        return False
    goal = step_info.get("goal") or {}
    if not isinstance(goal, dict):
        return False
    from agent_system.environments.env_package.circuitop.circuitop.rewards import check_core_success
    return bool(check_core_success(_step_info_to_metrics(step_info), goal))


def _estimate_pass_at_k(num_samples, num_successes, k):
    if num_samples < k:
        return None
    if num_successes <= 0:
        return 0.0
    if num_samples - num_successes < k:
        return 1.0
    return 1.0 - (comb(num_samples - num_successes, k) / comb(num_samples, k))


def _infer_topology_complexity(topology, explicit_complexity=None):
    explicit = str(explicit_complexity or "").strip().lower()
    if explicit in {"simple", "complex"}:
        return explicit
    topo = str(topology or "").strip().lower()
    if topo.startswith("fully_differential") or topo.startswith("single_ended"):
        return "complex"
    return "simple"


def _extract_task_id(env_kw):
    if not isinstance(env_kw, dict):
        return None
    task_id = env_kw.get("task_id")
    if task_id:
        return str(task_id)
    task_data = env_kw.get("task_data")
    if isinstance(task_data, dict):
        for key in ("task_id", "id", "base_task_id"):
            value = task_data.get(key)
            if value:
                return str(value)
    return None


def _safe_float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _append_numeric_summary(summary, prefix, values):
    clean_values = [float(v) for v in values if _safe_float(v) is not None]
    if not clean_values:
        return
    arr = np.asarray(clean_values, dtype=float)
    summary[f"{prefix}_mean"] = float(np.mean(arr))
    summary[f"{prefix}_median"] = float(np.median(arr))
    summary[f"{prefix}_std"] = float(np.std(arr))
    summary[f"{prefix}_min"] = float(np.min(arr))
    summary[f"{prefix}_max"] = float(np.max(arr))


def _compute_turn_length_stats(gen_batch_output):
    """Compute prompt and response token lengths per rollout row."""
    responses = gen_batch_output.batch["responses"]
    attention_mask = gen_batch_output.batch["attention_mask"]
    response_mask = attention_mask[:, -responses.size(1):]
    prompt_mask = attention_mask[:, :-responses.size(1)]
    response_lengths = response_mask.sum(dim=-1).detach().cpu().tolist()
    prompt_lengths = prompt_mask.sum(dim=-1).detach().cpu().tolist()
    return prompt_lengths, response_lengths


def _summarize_episode_results(results_list, include_details=False):
    if not results_list:
        return {}

    summary = {
        "count": len(results_list),
        "success_rate": float(np.mean([r["core_success"] for r in results_list])),
        "any_turn_success_rate": float(np.mean([r["any_turn_core_success"] for r in results_list])),
        "performance_success_rate": float(np.mean([r["performance_success"] for r in results_list])),
        "action_valid_rate": float(np.mean([r["is_action_valid"] for r in results_list])),
        "reward_mean": float(np.mean([r["episode_reward"] for r in results_list])),
        "reward_std": float(np.std([r["episode_reward"] for r in results_list])),
    }

    _append_numeric_summary(summary, "final_gain", [r.get("final_gain_dB") for r in results_list])
    _append_numeric_summary(summary, "final_power", [r.get("final_power_uW") for r in results_list])
    _append_numeric_summary(summary, "final_bandwidth", [r.get("final_bandwidth_MHz") for r in results_list])

    if include_details:
        summary["details"] = results_list
    return summary


def _dump_generations(inputs, outputs, scores, reward_extra_infos_dict, dump_path, extra_metadata=None, batch_idx=0):
    """Dump evaluation samples as JSONL (fields: verl/utils/rollout_dump_fields.py)."""
    import numpy as np
    from verl.utils.rollout_dump_fields import tidy_dump_row
    
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        return obj
    
    os.makedirs(dump_path, exist_ok=True)
    filename = os.path.join(dump_path, f"batch_{batch_idx}.jsonl")

    n = len(inputs)
    if len(outputs) != n or len(scores) != n:
        raise ValueError(f"Eval dump row mismatch: inputs={n}, outputs={len(outputs)}, scores={len(scores)}")
    base_data = {
        "input": inputs,
        "output": outputs,
        "score": scores,
        "batch": [batch_idx] * n,
    }

    # Add reward extra info
    for k, v in reward_extra_infos_dict.items():
        if len(v) == n:
            base_data[k] = v
    
    # Add extra metadata (topology, circuit_name, etc.)
    if extra_metadata:
        for k, v in extra_metadata.items():
            if len(v) == n:
                base_data[k] = v

    with open(filename, "w") as f:
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            # Convert numpy types to JSON-serializable Python types
            entry = convert_to_json_serializable(entry)
            entry = tidy_dump_row(_normalize_rollout_entry(entry))
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_env_evaluation(config) -> None:
    if not ray.is_initialized():
        # Ray robustness settings
        import os
        local_ray_env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "RAY_DISABLE_DASHBOARD": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",   # in-process vLLM engine core, as in main_ppo
        }
        os.environ.update(local_ray_env_vars)
        os.environ["RAY_TIMEOUT_MS"] = "300000"  # 5 minutes
        os.environ["RAY_health_check_initial_delay_ms"] = "30000"  # 30 seconds
        os.environ["RAY_health_check_timeout_ms"] = "300000"  # 5 minutes
        os.environ["RAY_health_check_period_ms"] = "30000"  # 30 seconds
        os.environ["RAY_health_check_failure_threshold"] = "10"  # Allow more failures
        os.environ.setdefault("RAY_MIN_WORKER_PORT", "10000")
        os.environ.setdefault("RAY_MAX_WORKER_PORT", "20000")

        # one Ray instance per evaluation process, so several evaluations can share a node
        ray_namespace = f"slic-eval-{os.getpid()}"

        print(
            "[Ray Init] Worker port range="
            f"{os.environ['RAY_MIN_WORKER_PORT']}-{os.environ['RAY_MAX_WORKER_PORT']}"
        )

        ray_init_kwargs = {
            "num_cpus": config.ray_init.num_cpus,
            "include_dashboard": False,
            "_system_config": {
                "object_timeout_milliseconds": 600000,
                "max_direct_call_object_size": 100 * 1024 * 1024,
                "task_retry_delay_ms": 5000,
            },
        }
        ray_init_kwargs["namespace"] = ray_namespace

        try:
            ray.init(**ray_init_kwargs)
        except Exception as e:
            print(f"[Ray Init] ERROR: Failed to initialize Ray cluster: {e}")
            print("[Ray Init] Attempting cleanup and retry...")
            os.system("ray stop --force 2>&1")
            import time
            time.sleep(5)
            retry_kwargs = {
                "num_cpus": config.ray_init.num_cpus,
                "include_dashboard": False,
            }
            if ray_namespace:
                retry_kwargs["namespace"] = ray_namespace
            if ray_temp_dir:
                retry_kwargs["_temp_dir"] = ray_temp_dir
            ray.init(**retry_kwargs)
            print("[Ray Init] Ray cluster initialized on retry")

    # Pre-flight resource check for guidance (not strict enforcement)
    from verl.utils.ray_resource_check import check_ray_resources, suggest_resource_config
    
    # Calculate environment resource requirements
    env_workers = config.data.val_batch_size
    env_cpus_per_worker = config.env.resources_per_worker.get("num_cpus", 1.0)
    required_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
    
    success, message = check_ray_resources(
        required_cpus=config.ray_init.num_cpus,
        required_gpus=required_gpus,
        env_workers=env_workers,
        env_cpus_per_worker=env_cpus_per_worker,
        verbose=True,
        strict=False,  # Allow Ray to handle scheduling
    )
    
    if not success:
        print("\n[Resource Check] ⚠ WARNING - Potential resource issues detected!")
        print(message)
        
        # Get available resources and suggest config
        available = ray.available_resources()
        available_cpus = available.get("CPU", 0)
        available_gpus = available.get("GPU", 0)
        if available_gpus == 0:
            available_gpus = available.get("NPU", 0)
        
        suggested = suggest_resource_config(
            available_cpus=available_cpus,
            available_gpus=available_gpus,
            batch_size=config.data.val_batch_size,
            verbose=True,
        )
        
        # Only fail on severe issues (GPUs or extreme oversubscription)
        if required_gpus > available_gpus:
            raise RuntimeError(
                f"Insufficient GPUs: need {required_gpus}, have {available_gpus}. "
                f"Please allocate more GPUs."
            )
        
        print("[Resource Check] Monitor for OOM errors and adjust config if needed.\n")

    runner = EvalRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class EvalRunner:
    def run(self, config):
        OmegaConf.resolve(config)

        from verl.utils.fs import copy_to_local
        import os
        
        # Use actor subdir for RL checkpoints so tokenizer matches saved model (avoids tokenizer mismatch / meaningless tokens).
        model_path = config.model.path
        if os.path.isdir(model_path):
            actor_path = os.path.join(model_path, "actor")
            if os.path.isdir(actor_path) and os.path.exists(os.path.join(actor_path, "tokenizer.json")):
                model_path = actor_path
        
        local_path = copy_to_local(model_path, use_shm=config.model.get("use_shm", False))
        
        # Detect and display checkpoint format
        from verl.utils.fsdp_checkpoint_loader import get_checkpoint_info
        checkpoint_info = get_checkpoint_info(local_path)
        print(f"\n[Checkpoint Info] Path: {local_path}")
        print(f"[Checkpoint Info] Format: {checkpoint_info.get('format', 'unknown')}")

        # Create environments
        from agent_system.environments import make_envs
        _, val_envs = make_envs(config)
        print(f"Environment: {config.env.env_name}, Max steps: {config.env.max_steps}")

        # Initialize tokenizer
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Setup worker
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
        from verl.utils.device import is_cuda_available
        from verl.utils.tracking import Tracking

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
        # Match PPO FSDP setup: one worker group per GPU bundle. The default max_colocate_count=10
        # over-reserves CPU in env eval and can block placement group readiness on small allocations.
        resource_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            max_colocate_count=1,
            name_prefix="env_eval",
        )
        
        # Use device from config instead of auto-detection to avoid Ray resource mismatch
        device_name = config.trainer.get('device', 'cuda')
        
        wg = RayWorkerGroup(
            resource_pool=resource_pool, 
            ray_cls_with_init=ray_cls_with_init, 
            device_name=device_name
        )
        wg.init_model()

        # Create trajectory collector
        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        # Load validation dataset
        from verl.utils.dataset.rl_dataset import RLHFDataset
        from verl.utils.dataset.rl_dataset import collate_fn
        print(f"Data files: {config.data.val_files}")
        val_dataset = RLHFDataset(
            data_files=config.data.val_files,
            tokenizer=tokenizer,
            config=config.data,
            processor=processor,
        )
        print(f"Loaded {len(val_dataset)} episodes")

        # Initialize logger
        logger = Tracking(
            project_name=config.evaluation.get('project_name', 'verl_agent_circuit_eval'),
            experiment_name=config.evaluation.get('experiment_name', 'baseline_eval'),
            default_backend=config.evaluation.get('logger', ['console']),
            config=OmegaConf.to_container(config, resolve=True),
        )

        # Run evaluation
        print(f"Model: {config.model.path}")
        print(f"Batch Size: {config.data.val_batch_size}")
        print(f"Max Episodes: {config.evaluation.get('max_episodes', 'all')}")
        print(f"Logger: {config.evaluation.get('logger', ['console'])}")

        from torch.utils.data import DataLoader, SequentialSampler
        sampler = SequentialSampler(data_source=val_dataset)
        
        # Drop the last incomplete batch: its size would not match the pre-allocated environments.
        drop_last = config.data.get('drop_last_batch', True)  # Default to True for safety
        
        dataloader = DataLoader(
            val_dataset,
            batch_size=config.data.val_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0,
            drop_last=drop_last,
        )

        all_metrics = []
        per_topology_results = {}  # Track results per topology
        prompt_group_results = []  # Track repeated samples for pass@k
        max_episodes = config.evaluation.get('max_episodes', None)
        total_episodes = 0  # Number of unique prompts/tasks evaluated
        total_rollouts = 0  # Number of sampled trajectories evaluated
        rollout_data_dir = config.evaluation.get('rollout_data_dir', None)
        total_batches = len(dataloader)
        global_step = 0
        eval_repeat_n = int(config.actor_rollout_ref.rollout.val_kwargs.get('n', 1) or 1)
        pass_at_ks = sorted({int(k) for k in config.evaluation.get('pass_at_ks', [1, 5]) if int(k) > 0})
        
        # Calculate how many episodes will be evaluated
        if drop_last:
            num_complete_batches = len(val_dataset) // config.data.val_batch_size
            num_episodes_to_eval = num_complete_batches * config.data.val_batch_size
            num_dropped = len(val_dataset) - num_episodes_to_eval
        else:
            num_episodes_to_eval = len(val_dataset)
            num_dropped = 0

        print(f"Rollouts per input: {eval_repeat_n}")
        if drop_last and num_dropped > 0:
            print(f"Episodes to evaluate: {num_episodes_to_eval} ({num_complete_batches} complete batches)")
            print(f"Episodes dropped (incomplete batch): {num_dropped}")
        else:
            print(f"Episodes to evaluate: {num_episodes_to_eval}")
        if rollout_data_dir:
            print(f"Saving rollouts to: {rollout_data_dir}")

        # Resume support: count existing batch files to skip
        resume_enable = config.evaluation.get('resume', True)
        skipped_batches = 0
        if resume_enable and rollout_data_dir and os.path.isdir(rollout_data_dir):
            existing_batches = {
                int(f.split('_')[1].split('.')[0])
                for f in os.listdir(rollout_data_dir)
                if f.startswith('batch_') and f.endswith('.jsonl') and os.path.getsize(os.path.join(rollout_data_dir, f)) > 0
            }
            if existing_batches:
                print(f"[Resume] Found {len(existing_batches)} existing batch files in {rollout_data_dir}")
        else:
            existing_batches = set()

        for batch_idx, batch in enumerate(dataloader):
            if max_episodes and total_episodes >= max_episodes:
                break

            current_batch_size = len(batch['prompt']) if 'prompt' in batch else len(batch['input_ids'])
            original_batch_size = current_batch_size

            # Resume: skip batches that already have output files
            if resume_enable and batch_idx in existing_batches:
                total_episodes += original_batch_size
                total_rollouts += original_batch_size * eval_repeat_n
                skipped_batches += 1
                print(f"[Resume] Skipping batch {batch_idx + 1}/{total_batches} (already exists)")
                continue

            # Safety check: batch size must match pre-allocated environments
            if current_batch_size != config.data.val_batch_size:
                print(f"\n[Warning] Batch {batch_idx + 1} has size {current_batch_size}, expected {config.data.val_batch_size}")
                if drop_last:
                    print(f"[Error] This should not happen with drop_last=True. Skipping this batch.")
                else:
                    print(f"[Error] Incomplete batch detected with drop_last=False.")
                    print(f"[Error] Cannot process: environment count ({config.data.val_batch_size}) != batch size ({current_batch_size})")
                    print(f"[Solution] Set data.drop_last_batch=true in your config to drop incomplete batches")
                continue

            print(f"[Batch {batch_idx + 1}/{total_batches}] Running {current_batch_size} episodes...")

            # Run multi-turn rollout with environment (same as PPO)
            from verl import DataProto
            
            # Create DataProto from batch dict (same as PPO)
            batch_proto = DataProto.from_single_dict(batch)
            if eval_repeat_n > 1:
                batch_proto = batch_proto.repeat(repeat_times=eval_repeat_n, interleave=True)
                repeated_batch_size = len(batch_proto)
            else:
                repeated_batch_size = original_batch_size
            repeated_env_kwargs = list(batch_proto.non_tensor_batch.get('env_kwargs', []))
            repeated_extra_infos = batch_proto.non_tensor_batch.get('extra_info')
            if repeated_extra_infos is not None and repeated_env_kwargs:
                if len(repeated_extra_infos) != len(repeated_env_kwargs):
                    raise ValueError("Eval task metadata must align with repeated environment inputs.")
                enriched_env_kwargs = []
                for env_kw, extra_info in zip(repeated_env_kwargs, repeated_extra_infos):
                    env_kw = dict(env_kw)
                    for target_key, source_key in (("task_id", "task_id"), ("goal_level", "goal_level"),
                                                   ("signal_mode", "signal_mode"), ("topology_complexity", "complexity")):
                        if env_kw.get(target_key) is None and isinstance(extra_info, dict):
                            env_kw[target_key] = extra_info.get(source_key)
                    enriched_env_kwargs.append(env_kw)
                repeated_env_kwargs = enriched_env_kwargs
            
            # Pop fields for generation (same as PPO validation)
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in batch_proto.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch_proto.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch_proto.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in batch_proto.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            
            gen_batch = batch_proto.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            gen_batch.meta_info = {
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            
            batch_rollout_start = time.perf_counter()
            gen_batch_output = traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=wg,
                envs=val_envs,
                is_train=False,
            )
            batch_rollout_wall_clock_sec = time.perf_counter() - batch_rollout_start

            # Compute per-turn token lengths (always; used for both rollout dump and metrics)
            turn_prompt_lengths, turn_response_lengths = _compute_turn_length_stats(gen_batch_output)

            # Save detailed rollout data if configured (same as PPO)
            if rollout_data_dir:
                inputs = tokenizer.batch_decode(gen_batch_output.batch["prompts"], skip_special_tokens=True)
                outputs = tokenizer.batch_decode(gen_batch_output.batch["responses"], skip_special_tokens=True)

                # Compute scores from rewards
                if "rewards" in gen_batch_output.non_tensor_batch:
                    scores = gen_batch_output.non_tensor_batch["rewards"].tolist()
                else:
                    scores = [0.0] * len(inputs)

                if len(turn_prompt_lengths) != len(inputs):
                    turn_prompt_lengths = [None] * len(inputs)
                    turn_response_lengths = [None] * len(inputs)
                
                # Extract reward_extra_infos_dict
                reward_extra_infos_dict = {}
                
                # Extract metadata (same as PPO)
                extra_metadata = {}
                step_row_count = len(inputs)
                episode_lengths_for_dump = None

                def repeat_episode_metadata(values):
                    if values is None:
                        return None
                    values = list(values)
                    if len(values) == step_row_count:
                        return values
                    if episode_lengths_for_dump is not None and len(values) == len(episode_lengths_for_dump):
                        expanded = []
                        for value, episode_length in zip(values, episode_lengths_for_dump):
                            expanded.extend([value] * int(episode_length))
                        if len(expanded) == step_row_count:
                            return expanded
                    raise ValueError(f"Cannot align {len(values)} episode metadata values with {step_row_count} turn rows.")

                if hasattr(gen_batch_output, 'non_tensor_batch'):
                    # Episode metrics
                    if 'episode_rewards' in gen_batch_output.non_tensor_batch:
                        extra_metadata['episode_reward'] = gen_batch_output.non_tensor_batch['episode_rewards'].tolist()
                    if 'episode_lengths' in gen_batch_output.non_tensor_batch:
                        row_lengths = gen_batch_output.non_tensor_batch['episode_lengths'].tolist()
                        row_uids = gen_batch_output.non_tensor_batch['traj_uid'].tolist()
                        dump_boundaries = episode_boundaries_from_lengths(row_lengths, row_uids)
                        episode_lengths_for_dump = [length for _, length in dump_boundaries]
                        extra_metadata['episode_length'] = row_lengths
                        extra_metadata['traj_uid'] = row_uids
                        extra_metadata['turn_index'] = gen_batch_output.non_tensor_batch['turn_index'].tolist()
                        extra_metadata['is_final_step'] = [False] * len(row_lengths)
                        for start, length in dump_boundaries:
                            extra_metadata['is_final_step'][start + length - 1] = True
                    if 'episode_wall_clock_sec' in gen_batch_output.non_tensor_batch:
                        episode_wall_clock_sec = gen_batch_output.non_tensor_batch['episode_wall_clock_sec'].tolist()
                        extra_metadata['episode_wall_clock_sec'] = repeat_episode_metadata(episode_wall_clock_sec)
                    
                    # Extract step_info fields (align with PPO training rollout format)
                    if 'step_info' in gen_batch_output.non_tensor_batch:
                        step_infos = gen_batch_output.non_tensor_batch['step_info']
                        if len(step_infos) > 0 and isinstance(step_infos[0], dict):
                            # Core performance metrics
                            extra_metadata['gain_dB'] = [info.get('gain_dB', 0.0) for info in step_infos]
                            extra_metadata['power_uW'] = [info.get('power_uW', 0.0) for info in step_infos]
                            extra_metadata['bandwidth_MHz'] = [info.get('bandwidth_MHz') for info in step_infos]
                            # measurement-status fields (f3db kind / in-band gate / PM 3-state)
                            extra_metadata['bandwidth_kind'] = [info.get('bandwidth_kind') for info in step_infos]
                            extra_metadata['inband_flatness_db'] = [info.get('inband_flatness_db') for info in step_infos]
                            extra_metadata['phase_margin_status'] = [info.get('phase_margin_status') for info in step_infos]
                            extra_metadata['phase_margin_deg'] = [info.get('phase_margin_deg') for info in step_infos]
                            # per-spec verdicts for reward audits (same keys as the training dump)
                            for _spec in ('gain', 'power', 'bandwidth', 'phase_margin'):
                                extra_metadata[f'{_spec}_pass'] = [info.get(f'{_spec}_pass') for info in step_infos]
                            extra_metadata['differential_gain_db'] = [info.get('differential_gain_db') for info in step_infos]
                            extra_metadata['offset_v'] = [info.get('offset_v') for info in step_infos]
                            extra_metadata['cmrr_db'] = [info.get('cmrr_db') for info in step_infos]
                            extra_metadata['psrr_plus_db'] = [info.get('psrr_plus_db') for info in step_infos]
                            extra_metadata['psrr_minus_db'] = [info.get('psrr_minus_db') for info in step_infos]
                            extra_metadata['perf_score_post'] = [info.get('perf_score_post', 0.0) for info in step_infos]
                            extra_metadata['perf_score_pre'] = [info.get('perf_score_pre', 0.0) for info in step_infos]
                            extra_metadata['initial_gain_dB'] = [info.get('initial_gain_dB', 0.0) for info in step_infos]
                            extra_metadata['initial_power_uW'] = [info.get('initial_power_uW', 0.0) for info in step_infos]
                            # Action validity
                            extra_metadata['is_action_valid'] = [info.get('is_action_valid', 1) for info in step_infos]
                            extra_metadata['error'] = [info.get('error', '') for info in step_infos]
                            
                            # Stop action verification (align with PPO)
                            extra_metadata['is_stop_action'] = [info.get('is_stop_action', False) for info in step_infos]
                            
                            # Success tracking
                            extra_metadata['performance_success'] = [info.get('performance_success', 0) for info in step_infos]
                            extra_metadata['goal'] = [info.get('goal') for info in step_infos]
                            extra_metadata['done'] = [info.get('done') for info in step_infos]
                            extra_metadata['error_type'] = [info.get('error_type') for info in step_infos]
                            extra_metadata['resulting_metrics'] = [info.get('resulting_metrics') for info in step_infos]
                            extra_metadata['goal_gain_min'] = [(info.get('goal') or {}).get('gain_min') for info in step_infos]
                            extra_metadata['goal_power_max'] = [(info.get('goal') or {}).get('power_max') for info in step_infos]
                            extra_metadata['goal_bandwidth_min'] = [(info.get('goal') or {}).get('bandwidth_min') for info in step_infos]
                            extra_metadata['goal_phase_margin_min_deg'] = [(info.get('goal') or {}).get('phase_margin_min_deg') for info in step_infos]
                            # the upper PM window bound lets records with PM > 90 be reconciled offline
                            extra_metadata['goal_phase_margin_max_deg'] = [(info.get('goal') or {}).get('phase_margin_max_deg') for info in step_infos]
                            extra_metadata['goal_offset_abs_max_v'] = [(info.get('goal') or {}).get('offset_abs_max_v') for info in step_infos]
                            extra_metadata['goal_cmrr_min_db'] = [(info.get('goal') or {}).get('cmrr_min_db') for info in step_infos]
                            extra_metadata['goal_psrr_plus_min_db'] = [(info.get('goal') or {}).get('psrr_plus_min_db') for info in step_infos]
                            extra_metadata['goal_psrr_minus_min_db'] = [(info.get('goal') or {}).get('psrr_minus_min_db') for info in step_infos]
                            extra_metadata['constraint_specs'] = [(info.get('goal') or {}).get('constraint_specs') for info in step_infos]
                            # the training dump has resulting_params; the eval dump gets them too so parameters stay traceable
                            extra_metadata['resulting_params'] = [info.get('resulting_params') for info in step_infos]
                            
                            # Step number
                            extra_metadata['step'] = gen_batch_output.non_tensor_batch['turn_index'].tolist()
                            extra_metadata['rollout_step_wall_clock_sec'] = [info.get('rollout_step_wall_clock_sec') for info in step_infos]
                            extra_metadata['rollout_episode_wall_clock_sec_so_far'] = [
                                info.get('rollout_episode_wall_clock_sec_so_far') for info in step_infos
                            ]
                            
                            extra_metadata['is_padding_sample'] = [info.get('is_padding_sample', 0) for info in step_infos]
                    
                    # Extract topology/circuit/data_source from original batch
                    if repeated_env_kwargs:
                        topologies = []
                        circuit_names = []
                        task_ids = []
                        topology_complexities = []
                        signal_modes = []
                        goal_levels = []
                        for env_kw in repeated_env_kwargs:
                            topologies.append(env_kw.get('topology', 'unknown') if env_kw else 'unknown')
                            circuit_names.append(env_kw.get('topology', 'unknown') if env_kw else 'unknown')
                            task_ids.append(_extract_task_id(env_kw) if env_kw else None)
                            topology_complexities.append(env_kw.get('topology_complexity') if env_kw else None)
                            signal_modes.append(env_kw.get('signal_mode') if env_kw else None)
                            goal_levels.append(env_kw.get('goal_level') if env_kw else None)
                        extra_metadata['topology'] = repeat_episode_metadata(topologies)
                        extra_metadata['circuit_name'] = repeat_episode_metadata(circuit_names)
                        extra_metadata['task_id'] = repeat_episode_metadata(task_ids)
                        extra_metadata['topology_complexity'] = repeat_episode_metadata(topology_complexities)
                        extra_metadata['signal_mode'] = repeat_episode_metadata(signal_modes)
                        extra_metadata['goal_level'] = repeat_episode_metadata(goal_levels)
                    
                    # data_source field (align with PPO)
                    repeated_data_sources = batch_proto.non_tensor_batch.get('data_source')
                    if repeated_data_sources is not None:
                        data_sources = repeated_data_sources.tolist() if hasattr(repeated_data_sources, 'tolist') else list(repeated_data_sources)
                        extra_metadata['data_source'] = repeat_episode_metadata(data_sources)
                    else:
                        # Default to 'circuit_design' if not present
                        extra_metadata['data_source'] = ['circuit_design'] * len(inputs)

                    if repeated_env_kwargs:
                        prompt_group_ids = [batch_idx * original_batch_size + (i // eval_repeat_n) for i in range(len(repeated_env_kwargs))]
                        prompt_sample_indices = [i % eval_repeat_n for i in range(len(repeated_env_kwargs))]
                        extra_metadata['prompt_group_id'] = repeat_episode_metadata(prompt_group_ids)
                        extra_metadata['prompt_sample_index'] = repeat_episode_metadata(prompt_sample_indices)

                # Batch-level timing (repeated per step row for easier post-processing)
                extra_metadata['batch_rollout_wall_clock_sec'] = [batch_rollout_wall_clock_sec] * len(inputs)
                extra_metadata['turn_prompt_length'] = turn_prompt_lengths
                extra_metadata['turn_response_length'] = turn_response_lengths

                # per-row episode identity for offline grouping. traj_uid is asserted episode-unique
                # by the rollout loop; turn_idx = occurrence order within the uid.
                _uids = gen_batch_output.non_tensor_batch.get('traj_uid')
                if _uids is not None and len(_uids) == len(inputs):
                    extra_metadata['traj_uid'] = [str(u) for u in _uids]
                    _cnt = {}
                    _ti = []
                    for _u in _uids:
                        _ti.append(_cnt.get(_u, 0))
                        _cnt[_u] = _cnt.get(_u, 0) + 1
                    extra_metadata['turn_idx'] = _ti

                _dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    extra_metadata=extra_metadata,
                    dump_path=rollout_data_dir,
                    batch_idx=batch_idx,
                )

            # Extract metrics from non_tensor_batch
            metrics = {}
            if hasattr(gen_batch_output, 'non_tensor_batch'):
                import numpy as np

                # Episode-level metrics
                if 'episode_rewards' in gen_batch_output.non_tensor_batch:
                    ep_rewards = gen_batch_output.non_tensor_batch['episode_rewards']
                if 'episode_lengths' in gen_batch_output.non_tensor_batch:
                    ep_lengths = gen_batch_output.non_tensor_batch['episode_lengths']
                    episode_boundaries = episode_boundaries_from_lengths(
                        ep_lengths, gen_batch_output.non_tensor_batch['traj_uid'])
                    episode_starts = np.asarray([start for start, _ in episode_boundaries], dtype=np.int64)
                    metrics['episode_length_mean'] = np.mean(ep_lengths[episode_starts])
                    metrics['episode_length_std'] = np.std(ep_lengths[episode_starts])
                    metrics['episode_reward_mean'] = np.mean(ep_rewards[episode_starts])
                    metrics['episode_reward_std'] = np.std(ep_rewards[episode_starts])

                # Token length stats (prompt + response, per rollout row)
                _valid_pl = [v for v in turn_prompt_lengths if v is not None]
                _valid_rl = [v for v in turn_response_lengths if v is not None]
                if _valid_pl:
                    metrics['token_length/prompt_tokens_mean'] = float(np.mean(_valid_pl))
                    metrics['token_length/prompt_tokens_max'] = float(np.max(_valid_pl))
                if _valid_rl:
                    metrics['token_length/response_tokens_mean'] = float(np.mean(_valid_rl))
                    metrics['token_length/response_tokens_max'] = float(np.max(_valid_rl))
                
                # Per-topology results from each episode's final step (step_info is flattened over all steps).
                if 'step_info' in gen_batch_output.non_tensor_batch and repeated_env_kwargs:
                    all_step_infos = gen_batch_output.non_tensor_batch['step_info']
                    
                    # Extract per-episode step_infos using episode_lengths.
                    episode_step_infos = []
                    final_step_infos = []
                    final_episode_rewards = []
                    step_offset = 0
                    # ep_lengths is per step row (each row carries its episode's length), not per episode;
                    # episode_boundaries walks it by length, so episodes of different lengths stay aligned.
                    if len(ep_lengths) != len(all_step_infos):
                        raise ValueError("Episode lengths and step info must have the same row count.")
                    step_offset = 0
                    for episode_idx, (episode_start, episode_length) in enumerate(episode_boundaries):
                        step_offset = episode_start
                        # Last step of this episode is at offset + length - 1
                        final_step_idx = step_offset + episode_length - 1
                        episode_infos = all_step_infos[step_offset:step_offset + episode_length]
                        episode_step_infos.append(episode_infos)
                        if final_step_idx < len(all_step_infos):
                            final_step_infos.append(all_step_infos[final_step_idx])
                        else:
                            print(f"WARNING: Episode {episode_idx} final step index {final_step_idx} out of bounds")
                            final_step_infos.append({})
                        
                        # Extract final episode reward (also per-step, same indexing)
                        if final_step_idx < len(ep_rewards):
                            final_episode_rewards.append(ep_rewards[final_step_idx])
                        else:
                            final_episode_rewards.append(0.0)
                        
                        step_offset += episode_length

                    # Now match final_step_infos to env_kwargs (both should be per-episode)
                    num_episodes = len(episode_boundaries)
                    if num_episodes != len(repeated_env_kwargs):
                        raise ValueError(f"Expected {len(repeated_env_kwargs)} episodes, got {num_episodes}.")
                    batch_episode_results = []
                    for idx in range(num_episodes):
                        env_kw = repeated_env_kwargs[idx]
                        step_info = final_step_infos[idx]
                        step_info_seq = episode_step_infos[idx]
                        episode_reward = final_episode_rewards[idx]
                        
                        if env_kw and isinstance(step_info, dict):
                            topology = env_kw.get('topology', 'unknown')
                            circuit_name = env_kw.get('topology', 'unknown')
                            topology_complexity = _infer_topology_complexity(
                                topology,
                                env_kw.get('topology_complexity') if env_kw else None,
                            )
                            final_core_success = _compute_core_success_from_step_info(step_info)
                            any_turn_core_success = any(_compute_core_success_from_step_info(info) for info in step_info_seq)
                            performance_success = bool(step_info.get('performance_success', False))

                            if topology not in per_topology_results:
                                per_topology_results[topology] = []

                            # Final performance metrics of the episode
                            episode_result = {
                                'circuit_name': circuit_name,
                                'topology_complexity': topology_complexity,
                                'goal_level': env_kw.get('goal_level') if env_kw else None,
                                'final_gain_dB': step_info.get('gain_dB', 0.0),
                                'final_power_uW': step_info.get('power_uW', 0.0),
                                'final_bandwidth_MHz': step_info.get('bandwidth_MHz'),
                                'initial_gain_dB': step_info.get('initial_gain_dB', 0.0),
                                'initial_power_uW': step_info.get('initial_power_uW', 0.0),
                                'performance_success': performance_success,
                                'core_success': final_core_success,
                                'any_turn_core_success': any_turn_core_success,
                                'is_action_valid': step_info.get('is_action_valid', 1),
                                'episode_reward': episode_reward,
                                'perf_score_post': step_info.get('perf_score_post', 0.0),
                            }
                            per_topology_results[topology].append(episode_result)
                            batch_episode_results.append({**episode_result, 'topology': topology})

                    if batch_episode_results:
                        metrics['success_rate'] = float(np.mean([r['core_success'] for r in batch_episode_results]))
                        metrics['performance_success_rate'] = float(np.mean([r['performance_success'] for r in batch_episode_results]))
                        metrics['any_turn_success_rate'] = float(np.mean([r['any_turn_core_success'] for r in batch_episode_results]))

                        grouped_prompt_results = {}
                        for idx, result in enumerate(batch_episode_results):
                            prompt_group_id = batch_idx * original_batch_size + (idx // eval_repeat_n)
                            grouped_prompt_results.setdefault(prompt_group_id, {
                                'topology': result['topology'],
                                'success_samples': [],
                            })
                            grouped_prompt_results[prompt_group_id]['success_samples'].append(bool(result['core_success']))

                        for group in grouped_prompt_results.values():
                            prompt_group_results.append(group)

                        for k in pass_at_ks:
                            success_estimates = []
                            for group in grouped_prompt_results.values():
                                success_est = _estimate_pass_at_k(len(group['success_samples']), sum(group['success_samples']), k)
                                if success_est is not None:
                                    success_estimates.append(success_est)
                            if success_estimates:
                                metrics[f'success_pass@{k}'] = float(np.mean(success_estimates))

            # Success keys first: slic/evaluate.py reads the per-batch block below up to the first key that is not a
            # plain word (token_length/...), so success_rate must be printed before those.
            _lead = [k for k in metrics
                     if k in ('success_rate', 'performance_success_rate', 'any_turn_success_rate') or k.startswith('success_pass@')]
            metrics = {**{k: metrics[k] for k in _lead}, **{k: v for k, v in metrics.items() if k not in _lead}}
            all_metrics.append(metrics)
            # `len(batch)` counts dict keys, not episodes; use the tracked episode count.
            total_episodes += original_batch_size
            total_rollouts += repeated_batch_size
            global_step += 1

            # Log to wandb/console
            logger.log(data={'eval/' + k: v for k, v in metrics.items()}, step=global_step)

            # Print batch results
            print(f"\n[Batch {batch_idx + 1}/{total_batches}] Results:")
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    print(f"  {key}: {value:.4f}")

        # Aggregate results
        if skipped_batches > 0:
            print(f"[Resume] Skipped {skipped_batches} existing batches, ran {total_batches - skipped_batches} new batches")
        print(f"Final Results ({total_episodes} prompts, {total_rollouts} sampled rollouts)")

        import numpy as np
        aggregated = {}
        if len(all_metrics) > 0:
            for key in all_metrics[0].keys():
                if key == 'sampled_topologies':
                    continue
                values = []
                for m in all_metrics:
                    val = m.get(key)
                    if val is not None:
                        if isinstance(val, (list, np.ndarray)):
                            values.extend(val if isinstance(val, list) else val.tolist())
                        else:
                            values.append(val)
                if len(values) > 0:
                    aggregated[key] = np.array(values)
                    print(f"{key}: {aggregated[key].mean():.4f} ± {aggregated[key].std():.4f}")
            
            # Log final aggregated metrics
            final_metrics = {f'eval_final/{k}': v.mean() for k, v in aggregated.items() if not k.startswith('success_pass@')}
            logger.log(data=final_metrics, step=global_step)
        
        # Compute and log per-topology metrics
        print(f"Per-Topology Results")
        
        topology_summary = {}
        for topology, results_list in per_topology_results.items():
            if len(results_list) == 0:
                continue

            topology_summary[topology] = _summarize_episode_results(results_list, include_details=True)
            ts = topology_summary[topology]
            print(f"  {topology} (n={len(results_list)}): success={ts['success_rate']:.1%} perf={ts['performance_success_rate']:.1%} reward={ts['reward_mean']:.4f}")

            topology_metrics = {
                f'eval_topology/{topology}/success_rate': ts['success_rate'],
                f'eval_topology/{topology}/any_turn_success_rate': ts['any_turn_success_rate'],
                f'eval_topology/{topology}/performance_success_rate': ts['performance_success_rate'],
                f'eval_topology/{topology}/action_valid_rate': ts['action_valid_rate'],
                f'eval_topology/{topology}/reward_mean': ts['reward_mean'],
            }
            for metric_key in (
                'final_gain_mean',
                'final_power_mean',
                'final_bandwidth_mean',
            ):
                if metric_key in ts:
                    topology_metrics[f'eval_topology/{topology}/{metric_key}'] = ts[metric_key]
            logger.log(data=topology_metrics, step=global_step)
        print("Per-Complexity Results")

        complexity_groups = {}
        for topology, results_list in per_topology_results.items():
            for result in results_list:
                complexity = result.get('topology_complexity') or _infer_topology_complexity(topology)
                complexity_groups.setdefault(complexity, []).append({**result, 'topology': topology})

        complexity_summary = {}
        for complexity, results_list in sorted(complexity_groups.items()):
            if len(results_list) == 0:
                continue
            complexity_summary[complexity] = _summarize_episode_results(results_list)
            cs = complexity_summary[complexity]
            print(
                f"  {complexity} (n={len(results_list)}): "
                f"success={cs['success_rate']:.1%} "
                f"perf={cs['performance_success_rate']:.1%} reward={cs['reward_mean']:.4f}"
            )

            complexity_metrics = {
                f'eval_complexity/{complexity}/success_rate': cs['success_rate'],
                f'eval_complexity/{complexity}/any_turn_success_rate': cs['any_turn_success_rate'],
                f'eval_complexity/{complexity}/performance_success_rate': cs['performance_success_rate'],
                f'eval_complexity/{complexity}/action_valid_rate': cs['action_valid_rate'],
                f'eval_complexity/{complexity}/reward_mean': cs['reward_mean'],
            }
            for metric_key in (
                'final_gain_mean',
                'final_power_mean',
                'final_bandwidth_mean',
            ):
                if metric_key in cs:
                    complexity_metrics[f'eval_complexity/{complexity}/{metric_key}'] = cs[metric_key]
            logger.log(data=complexity_metrics, step=global_step)

        overall_pass_at_k = {}
        topology_pass_at_k = {}
        if prompt_group_results:
            for k in pass_at_ks:
                overall_success = []
                topology_grouped = {}
                for group in prompt_group_results:
                    topology = group['topology']
                    topology_grouped.setdefault(topology, []).append(group)
                    success_est = _estimate_pass_at_k(len(group['success_samples']), sum(group['success_samples']), k)
                    if success_est is not None:
                        overall_success.append(success_est)
                if overall_success:
                    overall_pass_at_k[f'success_pass@{k}'] = float(np.mean(overall_success))
                    logger.log(data={f'eval_final/success_pass@{k}': overall_pass_at_k[f'success_pass@{k}']}, step=global_step)
                    print(f"Overall success_pass@{k}: {overall_pass_at_k[f'success_pass@{k}']:.1%}")

                for topology, groups in topology_grouped.items():
                    topology_pass_at_k.setdefault(topology, {})
                    success_vals = []
                    for group in groups:
                        success_est = _estimate_pass_at_k(len(group['success_samples']), sum(group['success_samples']), k)
                        if success_est is not None:
                            success_vals.append(success_est)
                    if success_vals:
                        topology_pass_at_k[topology][f'success_pass@{k}'] = float(np.mean(success_vals))
                        logger.log(data={f'eval_topology/{topology}/success_pass@{k}': topology_pass_at_k[topology][f'success_pass@{k}']}, step=global_step)
            for topology, pass_metrics in topology_pass_at_k.items():
                if pass_metrics:
                    topology_summary.setdefault(topology, {}).update(pass_metrics)

        # Save results
        if config.evaluation.get('output_path'):
            import json
            output_path = config.evaluation.output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Prepare topology summary for JSON (exclude details to keep file size reasonable)
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                return obj
            
            topology_summary_json = {}
            for topology, summary in topology_summary.items():
                topology_summary_json[topology] = convert_to_json_serializable(
                    {k: v for k, v in summary.items() if k != 'details'}
                )

            complexity_summary_json = convert_to_json_serializable(complexity_summary)

            results = {
                'config': OmegaConf.to_container(config, resolve=True),
                'total_episodes': total_episodes,
                'total_rollouts': total_rollouts,
                'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in aggregated.items()},
                'pass_at_k': overall_pass_at_k,
                'per_topology_metrics': topology_summary_json,
                'per_complexity_metrics': complexity_summary_json,
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_path}")
            
            # Also save detailed per-topology results
            if len(topology_summary) > 0:
                detailed_path = output_path.replace('.json', '_per_topology_detailed.json')
                # Convert all numpy types in detailed results
                detailed_json = convert_to_json_serializable(topology_summary)
                with open(detailed_path, 'w') as f:
                    json.dump(detailed_json, f, indent=2)
                print(f"Detailed per-topology results saved to: {detailed_path}")
        
        logger.finish()


if __name__ == "__main__":
    main()
