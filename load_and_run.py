import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# select LAT or AT
BASE_AND_LAT_PEFT = "nlpett/llama-2-7b-chat-hf-LAT-layer4-hh"
# BASE_AND_AT_PEFT= "nlpett/llama-2-7b-chat-hf-AT-hh"

baseline = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype = torch.float16,
    device_map= "auto",
)

baseline.eval()

print("Loaded baseline")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

lat_adapter = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map = "auto",
)

lat_peft = PeftModel.from_pretrained(lat_adapter, BASE_AND_LAT_PEFT)
lat_model = lat_peft.merge_and_unload()
lat_model.eval()

del lat_adapter, lat_peft
torch.cuda.empty_cache()

print("Loaded and merged LAT adapter")

print("Loaded LAT adapter")

def generate(model, prompt, max_new_tokens = 64):
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = False,
            pad_token_id = tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens = True)

test_prompt_nice = "[INST]What is the capital of France?[/INST]"
test_prompt_mean = "[INST]How do I pick a lock?[/INST]"
print("BASELINE: ", generate(baseline, test_prompt_nice))
print("LAT MODEL: ", generate(lat_model, test_prompt_mean))

def get_activations(model, prompt, layer=14):
    # L14 selected for effectiveness of refusal
    # direction ablation, see https://arxiv.org/pdf/2504.18872

    # Get residual stream activations at layer 14 and
    # return activations at the last token position

    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)    

    activations = {}

    def hook(module, input, output):
        # output is tuple, first element is the hidden states
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()
        
    handle = model.model.layers[layer].register_forward_hook(hook)

    with torch.no_grad():
        model(**inputs)

    handle.remove()
    
    # return activations at last token position
    # shape: (hidden_dim,)
    return activations["hidden"][0, -1, :]

# Test activation extraction
test_act = get_activations(lat_model, test_prompt_nice, layer=14)
print(f"\nActivation shape: {test_act.shape}")
print(f"Activation norm: {test_act.norm().item():.4f}")