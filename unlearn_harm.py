# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
import argparse
import logging
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

# from transformers import get_scheduler
# from ctransformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    compute_kl,
    create_pku_dataloader_from_dataset,
    create_truthfulqa_dataloader,
    get_answer_loss,
    get_rand_ans_loss,
    get_truthfulQA_answers_plaintext,
    print_trainable_parameters,
)

from wrappers import UnlearningModelWrapper


torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def main(args) -> None:
    accelerator = Accelerator()
    device = accelerator.device
    model_wrapper = UnlearningModelWrapper(args.model_name, args.use_lora)
    pass
    # # If use LoRA.
    # if args.use_lora:
    #     peft_config = AdaLoraConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         inference_mode=False,
    #         r=32,
    #         lora_alpha=16,
    #         target_modules=["q_proj", "v_proj"],
    #     )
    #     model_wrapper.new_model = get_peft_model(model_wrapper.new_model, peft_config)

    # print_trainable_parameters(model_wrapper.new_model)

    # model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # tokenizer.pad_token = tokenizer.bos_token
    # tokenizer.pad_token = '[PAD]'
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load harmful data.
    train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    train_bad_loader = create_pku_dataloader_from_dataset(
        tokenizer, train_dataset, batch_size=args.batch_size
    )

    # Get normal data.
    train_normal_loader, _, _ = create_truthfulqa_dataloader(
        tokenizer, batch_size=args.batch_size
    )

    # Load normal answer used for random mismatch.
    normal_ans = get_truthfulQA_answers_plaintext()

    optimizer = AdamW(model_wrapper.new_model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    (
        model_wrapper,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model_wrapper, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )

    model_wrapper.new_model.train()

    # Reference model for computing KL.
    # pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)

    # Start unlearning.
    bad_loss = torch.zeros(1)
    idx = 0
    start_time = time.time()
    # Stop if bad loss is big enough or reaching max step.
    while -bad_loss.item() < args.max_bad_loss and idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            ############ GA on bad answer only. ############
            if args.bad_weight > 0:
                bad_loss = get_answer_loss(
                    "ga",
                    bad_batch,
                    model_wrapper.new_model,
                    tokenizer=tokenizer,
                    device=device,
                )

                if -bad_loss.item() >= args.max_bad_loss:
                    print(
                        f"=====================Stop at bad loss: {bad_loss:.2f}========================"
                    )
                    break
            else:
                bad_loss = 0.0

            ############ Random mismatch. ############
            if not args.no_mismatch and args.random_weight > 0:
                random_loss = get_rand_ans_loss(
                    bad_batch,
                    tokenizer,
                    normal_ans,
                    model_wrapper.new_model,
                    K=5,
                    device=device,
                )
            else:
                random_loss = 0.0

            ############ KL on normal samples. ############
            if args.normal_weight > 0:
                normal_loss = compute_kl(
                    model_wrapper.new_model.model,
                    model_wrapper.new_model,
                    normal_batch,
                    device,
                )
            else:
                normal_loss = 0.0

            # Final loss = bad loss + random smoothing + normal loss.
            loss = (
                args.bad_weight * bad_loss
                + args.random_weight * random_loss
                + args.normal_weight * normal_loss
            )

            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Print.
            stats = (
                f"batch: {idx}, "
                f"bad_loss: {-bad_loss:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
                f"random_loss: {random_loss:.2f}, "
            )
            logging.info(stats)
            print(stats)
            idx += 1

            if idx >= args.max_unlearn_steps:
                break

            # Save model.
            if (idx + 1) % args.save_every == 0:
                model_wrapper.new_model.save_pretrained(
                    args.model_save_dir, from_pt=True
                )
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    # if args.use_lora:
    #     model_wrapper.new_model = model_wrapper.new_model.merge_and_unload()

    # Save final model.
    model_wrapper.new_model.save_pretrained(args.model_save_dir, from_pt=True)
    logging.info("Unlearning finished")


def upload_to_hf(args):
    from huggingface_hub import HfApi

    api = HfApi()
    # api.create_repo(repo_id=repo_id, private=True)
    future = api.upload_folder(  # Upload in the background (non-blocking action)
        repo_id=args.hf_repo_id,
        folder_path=args.model_save_dir,
        run_as_future=True,
        commit_message=f"Unlearn Harm with the following arguments: \n{args}",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--no_mismatch", action="store_true")

    parser.add_argument("--use_lora", action="store_true")

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=1000,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--bad_weight", type=float, default=0.5, help="Weight on the bad loss."
    )
    parser.add_argument(
        "--random_weight",
        type=float,
        default=1,
        help="Weight on learning the random outputs.",
    )
    parser.add_argument(
        "--normal_weight",
        type=float,
        default=1,
        help="Weight on normal loss.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=2e-6, help="Unlearning LR.")
    parser.add_argument(
        "--max_bad_loss",
        type=float,
        default=100,
        help="Maximum loss on bad samples to terminate.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-1.3b",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/opt1.3b_unlearned",
        help="Directory to save model.",
    )
    parser.add_argument(
        "--save_every", type=int, default=500, help="How many steps to save model."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/default.log",
        help="Log file name",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="HuggingFace Repo ID",
    )
    args = parser.parse_args()

    logging.basicConfig(
        filename=args.log_file,
        filemode="w+",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d-%H-%M",
        level=logging.INFO,
    )
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    main(args)
    if args.hf_repo_id:
        upload_to_hf(args)
