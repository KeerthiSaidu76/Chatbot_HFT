from __future__ import annotations
import argparse, random, torch
from dataclasses import dataclass, field
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------- Model Loader ----------------
def load_generation_pipeline(model_id: str = "distilgpt2", device: int | None = None, torch_dtype: str | None = None):
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    dtype = None
    if torch_dtype:
        dmap = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dmap.get(torch_dtype.lower())
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if dtype is not None:
        model = model.to(dtype=dtype)
    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return gen, tokenizer

# ---------------- Sliding Window Memory ----------------
@dataclass
class SlidingWindowMemory:
    max_turns: int = 5
    history: List[Dict[str, str]] = field(default_factory=list)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"user": user_text, "assistant": assistant_text})
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_window(self) -> List[Dict[str, str]]:
        return list(self.history)

def build_prompt(user_input: str, memory: SlidingWindowMemory) -> str:
    
    lines = []
    for turn in memory.get_window():
        lines.append(f"User: {turn['user'].strip()}")
        lines.append(f"Assistant: {turn['assistant'].strip()}")
    lines.append(f"User: {user_input.strip()}")
    lines.append("Assistant:")
    return "\n".join(lines)

# ---------------- CLI Interface ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Local CLI chatbot using Hugging Face.")
    p.add_argument("--model_id", type=str, default="distilgpt2", help="Model ID from Hugging Face")
    p.add_argument("--max_turns", type=int, default=5, help="How many turns of history to remember")
    p.add_argument("--max_new_tokens", type=int, default=128, help="Maximum reply length")
    p.add_argument("--temperature", type=float, default=0.7, help="Creativity (lower = more focused)")
    p.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling")
    p.add_argument("--greedy", action="store_true", help="Disable randomness, greedy decoding")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--torch_dtype", type=str, default=None, choices=[None, "float16", "bfloat16", "float32"])
    p.add_argument("--device", type=int, default=None, help="0 for GPU, -1 for CPU (default: auto)")
    return p.parse_args()

def set_seed(seed):
    if seed is None: return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def postprocess_reply(prompt: str, full_text: str) -> str:
    gen = full_text[len(prompt):]
    for stop in ["\nUser:", "\n###"]:
        if stop in gen:
            gen = gen.split(stop)[0]
    return gen.strip()

def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"Loading model {args.model_id}...")
    gen, tok = load_generation_pipeline(args.model_id, args.device, args.torch_dtype)
    memory = SlidingWindowMemory(max_turns=args.max_turns)

    print("Ready. Type your messages. Type /exit to quit.\n")
    try:
        while True:
            try:
                user = input("You: ").strip()
            except EOFError:
                user = "/exit"
            if user.lower() == "/exit":
                print("Exiting chatbot. Goodbye!"); break
            if not user: continue

            prompt = build_prompt(user, memory)
            out = gen(prompt,
                      max_new_tokens=args.max_new_tokens,
                      do_sample=not args.greedy,
                      temperature=args.temperature,
                      top_p=args.top_p,
                      pad_token_id=tok.eos_token_id,
                      eos_token_id=tok.eos_token_id)
            reply = postprocess_reply(prompt, out[0]["generated_text"])
            print(f"Bot: {reply}\n")
            memory.add_turn(user, reply)
    except KeyboardInterrupt:
        print("\nExiting chatbot. Goodbye!")

if __name__ == "__main__":
    main()
