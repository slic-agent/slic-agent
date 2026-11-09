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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    from verl.utils.slic_groups import merge_slic_groups
    merge_slic_groups(config)
    if not ray.is_initialized():
        # this is for local ray cluster
        import os
        local_ray_env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            "RAY_DISABLE_DASHBOARD": "1",
            # vLLM's engine core runs in-process, so the FSDP -> vLLM weight sync reaches llm_engine.model_executor
            # directly (verl/workers/sharding_manager/fsdp_vllm.py)
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
        os.environ.update(local_ray_env_vars)
        # Respect per-job Ray worker port overrides from the launcher. Fall back to
        # the broad default range only when no job-specific setting is provided.
        os.environ.setdefault("RAY_MIN_WORKER_PORT", "10000")
        os.environ.setdefault("RAY_MAX_WORKER_PORT", "20000")
        
        # Ray robustness settings
        os.environ["RAY_TIMEOUT_MS"] = "300000"  # 5 minutes
        os.environ["RAY_health_check_initial_delay_ms"] = "30000"  # 30 seconds
        os.environ["RAY_health_check_timeout_ms"] = "300000"  # 5 minutes
        os.environ["RAY_health_check_period_ms"] = "30000"  # 30 seconds
        os.environ["RAY_health_check_failure_threshold"] = "10"  # Allow more failures
        
        ray_namespace = os.environ.get("RAY_NAMESPACE")
        ray_temp_dir = os.environ.get("RAY_TEMP_DIR")
        if ray_temp_dir:
            os.makedirs(ray_temp_dir, exist_ok=True)

        ray_init_kwargs = {
            "num_cpus": config.ray_init.num_cpus,
            "include_dashboard": False,  # Reduce overhead
            "_system_config": {
                "object_timeout_milliseconds": 600000,  # 10 minutes
                "max_direct_call_object_size": 100 * 1024 * 1024,  # 100 MB
                "task_retry_delay_ms": 5000,  # 5 seconds between retries
            },
        }
        
        print("[Ray Init] Starting Ray with worker port range "
              f"{os.environ['RAY_MIN_WORKER_PORT']}-{os.environ['RAY_MAX_WORKER_PORT']}")
        if ray_namespace:
            ray_init_kwargs["namespace"] = ray_namespace
        if ray_temp_dir:
            ray_init_kwargs["_temp_dir"] = ray_temp_dir
        
        try:
            ray.init(**ray_init_kwargs)
        except Exception as e:
            print(f"[Ray Init] ERROR: Failed to initialize Ray cluster: {e}")
            print("[Ray Init] Attempting cleanup and retry...")
            os.system("ray stop --force 2>&1")
            import time
            time.sleep(5)
            retry_kwargs = dict(ray_init_kwargs)
            retry_kwargs.pop("_system_config", None)
            ray.init(**retry_kwargs)
            print("[Ray Init] Ray cluster initialized on retry")

    # Pre-flight resource check for guidance (not strict enforcement)
    from verl.utils.ray_resource_check import check_ray_resources, suggest_resource_config
    
    # Calculate environment resource requirements from config
    env_workers = config.data.train_batch_size
    env_cpus_per_worker = config.env.resources_per_worker.get("num_cpus", 0.1)
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
            batch_size=config.data.train_batch_size,
            verbose=True,
        )
        
        print("[Resource Check] Monitor for OOM errors and adjust config if needed.\n")

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        from verl.utils.fs import resolve_ppo_checkpoint_tokenizer_path
        tokenizer_path = resolve_ppo_checkpoint_tokenizer_path(local_path)

        # instantiate tokenizer (before make_envs so we can compute effective val_batch_size)
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(tokenizer_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # Compute effective val_batch_size accounting for stratified sampling
        # so that make_envs creates the correct number of environment workers.
        val_k = config.data.get("val_stratified_val_k", None)
        test_k = config.data.get("val_stratified_test_k", None)
        if val_k is not None or test_k is not None:
            _val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
            from verl.utils.dataset.stratified_val_sampler import (
                SplitStratifiedCycleSampler,
                build_split_and_topology_labels,
            )
            split_labels, topo_labels = build_split_and_topology_labels(_val_dataset)
            samples_per_split = {}
            if val_k is not None:
                samples_per_split["validation"] = int(val_k)
            if test_k is not None:
                samples_per_split["test"] = int(test_k)
            _val_sampler = SplitStratifiedCycleSampler(
                split_labels=split_labels,
                topology_labels=topo_labels,
                samples_per_split=samples_per_split,
                default_k=3,
                seed=int(OmegaConf.select(config, "env.seed", default=42)),
                fixed=bool(config.data.get("val_stratified_fixed", False)),
            )
            effective_val_batch_size = len(_val_sampler)
            if effective_val_batch_size != config.data.val_batch_size:
                print(
                    f"[Val Env] Stratified sampler effective batch size: {effective_val_batch_size} "
                    f"(config val_batch_size was {config.data.val_batch_size})"
                )
                with open_dict(config):
                    config.data.val_batch_size = effective_val_batch_size
            del _val_dataset, _val_sampler

        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config)

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        elif reward_manager_name == 'step':
            from agent_system.reward_manager import StepRewardManager
            reward_manager_cls = StepRewardManager
        else:
            raise NotImplementedError

        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
