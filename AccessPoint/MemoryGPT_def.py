from util import *
from tqdm import tqdm
import torch
from mem0 import Memory
from types import SimpleNamespace

BASE_MEM_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memories",
            "path": "./chroma_db"
        }
    }
}

class MemoryGPTInstance():
    def __init__(self,
                 checkpoint_path,
                 model_instance,
                 tokenizer,
                 config_start=BASE_MEM_CONFIG,
                 device=None):
        self.checkpoint_path = checkpoint_path

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Set up memory
        mem_path = Path(self.checkpoint_path)

        if mem_path.exists() and mem_path.is_dir():
            config_start["vector_store"]["config"]["path"] = self.checkpoint_path
        else:
            raise FileNotFoundError(f"Directory not found: {mem_path}")

        self.memory = Memory.from_config(config_start)

        # Load HuggingFace model and tokenizer
        self.tokenizer = tokenizer
        self.model = model_instance

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def query_model(self, query,
                    temperature=1, top_p=1, max_tokens=600, num_return_sequences=1, system_prompt=None):
        # Retrieve relevant memories
        memories = self.memory.search(query, user_id=USER_ID)
        context = "\n".join([m['memory'] for m in memories['results']])

        # Build messages with memory context
        messages = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Include memory context in the user message
        if context:
            full_query = f"Relevant context about the user:\n{context}\n\nUser: {query}"
        else:
            full_query = query
        messages.append({"role": "user", "content": full_query})

        # Apply chat template if available, otherwise use simple format
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            prompt_parts = []
            for msg in messages:
                if msg["role"] == "system":
                    prompt_parts.append(f"System: {msg['content']}\n")
                elif msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}\n")
            prompt_parts.append("Assistant:")
            prompt = "".join(prompt_parts)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = len(inputs['input_ids'][0])

        # Generate response(s)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=num_return_sequences
            )

        # Decode response(s)
        responses = self.tokenizer.batch_decode(
            outputs[:, prompt_len:],
            skip_special_tokens=True
        )

        # Return in OpenAI format if num_return_sequences=1, else list
        if num_return_sequences == 1:
            # Wrap in OpenAI-like response format
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=responses[0]
                        )
                    )
                ]
            )
        return responses

