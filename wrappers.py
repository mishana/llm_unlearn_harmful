from peft import AdaLoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM
from time import sleep

from utils import print_trainable_parameters

class UnlearningModelWrapper(nn.Module):
    def __init__(self, model_name, use_lora):
        super(UnlearningModelWrapper, self).__init__()
        self.new_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

        if use_lora:
            peft_config = AdaLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=32,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
            )
            self.new_model = get_peft_model(self.new_model, peft_config)

        print_trainable_parameters(self.new_model)
        sleep(10)

        self.orig_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
