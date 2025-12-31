import sys
import torch
import dotenv
import math
import time

torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
sys.path.append("source/SpecForge")
root_path = "speculative-decode"
dotenv.load_dotenv(f"{root_path}/.env")

import scripts.train_eagle3 as eagle


parser = eagle.parse_args()

# specforge configs
args = parser.parse_args([
    "--target-model-path", "/mnt/c/Users/qpdbc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    "--draft-model-config", f"{root_path}/configs/qwen3-0.6b-eagle3.json",
    "--train-data-path", f"{root_path}/dataset/sharegpt_train_8192.jsonl",
    "--build-dataset-num-proc", "2",
    "--output-dir", f"{root_path}/outputs/qwen0.6-8b-eagle3-sharegpt",
    "--num-epochs","1",
    "--batch-size","1",
    "--learning-rate", "1e-4",
    "--max-length", "4096",
    "--chat-template", "qwen",
    "--cache-dir", f"{root_path}/cache",
    "--embedding-key", "model.embed_tokens.weight",
    "--tp-size", "1",
    "--target-model-backend", "sglang",
    ])

args.target_batch_size = 1

# sglang configs
args.sglang_attention_backend  = "flashinfer"
args.sglang_mem_fraction_static  = 0.4
args.sglang_context_length = 40960
args.sglang_enable_nccl_nvls  = False
args.sglang_enable_symm_mem  = False
args.sglang_enable_torch_compile = True
args.sglang_max_running_requests = 1
args.sglang_max_total_tokens = 40960

# initial 
## node
eagle.init_distributed(
    timeout=args.dist_timeout,
    tp_size=args.tp_size,
    sp_ring_size=args.sp_ring_size,
    sp_ulysses_size=args.sp_ulysses_size,
)

## models
draft_model_config, draft_model = eagle.build_draft_model(args)
target_model, processor = eagle.build_target_model(args, draft_model_config, True)

## dataset
train_dataloader, vocab_mapping_path, eval_dataloader = eagle.build_dataloaders(
    args, draft_model_config, processor
)
draft_model.load_vocab_mapping(vocab_mapping_path)

## optimizer
steps_per_epoch = math.ceil(len(train_dataloader) / args.draft_accumulation_steps)
args.total_steps = args.num_epochs * steps_per_epoch

eagle3_model = eagle.OnlineEagle3Model(
    draft_model=draft_model,
    length=args.ttt_length,
    attention_backend=args.attention_backend,
    )

eagle3_model = eagle.FSDP(
    eagle3_model,
    use_orig_params=True,
    mixed_precision=eagle.MixedPrecision(
        param_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    sharding_strategy=eagle.ShardingStrategy.SHARD_GRAD_OP,
    process_group=eagle.dist.group.WORLD,
    )

optimizer = eagle.BF16Optimizer(
    draft_model,
    lr=args.learning_rate,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio,
    total_steps=args.total_steps,
)

## main training
global_step = 0
start_epoch = 0
last_time = time.time()
is_online=True
records = []
for epoch in range(start_epoch, args.num_epochs):
    # Run training
    train_dataloader.sampler.set_epoch(epoch + 1)
    _ = draft_model.train()
    for data in train_dataloader:
        global_step += 1
        plosses, acces = eagle.run_forward(args, eagle3_model, data, target_model, is_online)
        eagle.run_backward_and_update(args, plosses, optimizer, global_step)
        # log training metrics
        time_per_step = time.time() - last_time
        last_time = time.time()
        avg_loss = sum(pl for pl in plosses) / len(plosses)
        avg_acc = sum(acces) / len(acces)
        records.extend([{
            "step": global_step,
            "loss": f"{avg_loss:.2f}",
            "acc": f"{avg_acc:.2f}",
            "time": f"{time_per_step:.2f}",
            }])
        if global_step % (args.log_interval) == 0:
            print(records[-1])
        # Save Checkpoints
        if global_step % args.save_interval == 0:
            # Save the model
            eagle.save_checkpoints(args, epoch, global_step, eagle3_model, optimizer)
        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break
    if args.max_num_steps is not None and global_step >= args.max_num_steps:
        break

eagle.destroy_distributed()
