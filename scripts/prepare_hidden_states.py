"""
This script will generate the hidden states for the dataset.
By generating hidden states in advance, we can avoid:
- the memory overhead of loading target model
- the latency overhead of generating hidden states for each request.
"""

import argparse
import hashlib
import os
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from specforge.modeling.target.eagle3_target_model import SGLangEagle3TargetModel
from sglang.srt.managers.schedule_batch import Req
from tqdm import tqdm
from transformers import AutoTokenizer

from specforge.data import build_eagle3_dataset
from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput
from specforge.utils import print_with_rank, rank_0_priority
from specforge.distributed import get_tp_group, init_distributed

@torch.no_grad()
def save_tensor(hidden_states_cpu):
    for idx, (hidden_states, batch_save_info) in enumerate(hidden_states_cpu):
        if idx % torch.distributed.get_world_size() != torch.distributed.get_rank():
            continue
        hidden_states_list, aux_hidden_states_list = hidden_states

        for hidden_state, aux_hidden_state, (data_point, output_file) in zip(
            hidden_states_list, aux_hidden_states_list, batch_save_info
        ):
            data_point["hidden_state"] = hidden_state.clone().unsqueeze(0).cpu()
            data_point["aux_hidden_state"] = (
                aux_hidden_state.clone().unsqueeze(0).cpu()
            )
            assert not torch.any(
                torch.isnan(data_point["hidden_state"])
            ), "hidden_state is expected to be non-nan"
            assert not torch.any(
                torch.isnan(data_point["aux_hidden_state"])
            ), "aux_hidden_state is expected to be non-nan"
            torch.save(data_point, output_file)


@torch.no_grad()
def generate_offline_data(args, generator: SGLangEagle3TargetModel, dataset: Dataset, batch_size: int = 64):
    MIN_FILE_SIZE = 100 * 1024
    # Prepare inputs for warm up
    batch_save_info = []
    group_size = 5000
    batch_inputs = []

    for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
        group_start = (idx // group_size) * group_size
        group_end = group_start + group_size
        grouped_subdir = f"rows_{group_start}-{group_end}"

        os.makedirs(f"{args.output_path}/{grouped_subdir}", exist_ok=True)
        output_file = f"{args.output_path}/{grouped_subdir}/data_{idx}.ckpt"
        if (
            os.path.exists(output_file)
            and os.path.getsize(output_file) > MIN_FILE_SIZE
        ):
            # we skip it if the file already exists and is larger than 100KB
            continue

        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]
        loss_mask = row["loss_mask"]

        batch_inputs.append(
            (
                [input_ids, attention_mask, loss_mask],
                output_file,
            )
        )

        if len(batch_inputs) >= batch_size or idx == len(dataset) - 1:
            # when this is the last batch or the batch is full, we generate and save the data
            _, _, aux_hidden_states_list, last_hidden_states_list = generator.extend(
                input_ids=input_ids,
                attention_mask=attention_mask,
                loss_mask=loss_mask,
                return_last_hidden_states=True,
                return_logits=False
            )

            for ((input_ids, attention_mask, loss_mask), output_file), aux_hidden_states, last_hidden_states in zip(batch_inputs, aux_hidden_states_list, last_hidden_states_list):
                data_point = {
                    "input_ids": input_ids.squeeze(0),
                    "loss_mask": loss_mask.squeeze(0),
                    "aux_hidden_state": aux_hidden_states.unsqueeze(0).cpu(),
                    "hidden_state": last_hidden_states.unsqueeze(0).cpu(),
                }
                torch.save(data_point, output_file)
            torch.cuda.empty_cache()
            batch_inputs = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--chat-template", type=str, default="llama3")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dist-timeout", type=int, default=20)

    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--enable-aux-hidden-states", action="store_true")
    parser.add_argument("--aux-hidden-states-layers", type=str, default=None)
    parser.add_argument("--build-dataset-num-proc", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: support DP for hidden states generation
    tp_size = int(os.environ.get("WORLD_SIZE", "1"))
    init_distributed(tp_size=tp_size, timeout=args.dist_timeout)

    if args.aux_hidden_states_layers is not None:
        args.aux_hidden_states_layers = [
            int(layer) for layer in args.aux_hidden_states_layers.split(",")
        ]

    assert os.path.exists(
        args.data_path
    ), f"Dataset path {args.data_path} does not exist"
    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        args.output_path = root_path.joinpath("cache", "hidden_states")

    dataset = load_dataset("json", data_files=args.data_path)["train"]
    if args.num_samples is not None:
        dataset = dataset.select(range(args.num_samples))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    cache_params_string = (
        f"{args.data_path}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.model_path}"  # Tokenizer may also different
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    with rank_0_priority():
        eagle3_dataset = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
            cache_key=cache_key,
            num_proc=args.build_dataset_num_proc,
        )
        print_with_rank("Built dataset")

    hidden_states_generator = SGLangEagle3TargetModel.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        torch_dtype=torch.bfloat16,
        device="cuda",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    hidden_states_generator.set_aux_hidden_states_layers(args.aux_hidden_states_layers)
    generate_offline_data(args, hidden_states_generator, eagle3_dataset)


if __name__ == "__main__":
    main()
