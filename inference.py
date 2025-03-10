import json

from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer

model_base_dir = "./mistral_models"
model_dir = f"{model_base_dir}/mistral-7b-instruct-v0.3"
trained_mode_dir = f"{model_base_dir}/mistral-7b-instruct-v0.3_trained/checkpoints/checkpoint_000300/consolidated"

tokenizer = MistralTokenizer.from_file(f"{trained_mode_dir}/tokenizer.model.v3")  # change to extracted tokenizer file
model = Transformer.from_folder(f"{model_dir}")  # change to extracted model dir
model.load_lora(f"{trained_mode_dir}/lora.safetensors")

completion_request = ChatCompletionRequest(
    messages=[
        UserMessage(
            content="Create a full workflow JSON action for this instruction: Filter the 'temperatures' table to include only values greater than 36."
        )
    ]
)

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate(
    [tokens],
    model,
    max_tokens=64,
    temperature=0.0,
    eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])


r = json.loads(result)
print(json.dumps(r, indent=2))
