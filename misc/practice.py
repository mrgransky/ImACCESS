import huggingface_hub
import os
hf_tk: str = os.getenv("HUGGINGFACE_TOKEN")

info = huggingface_hub.model_info("Qwen/Qwen3-VL-2B-Instruct", token=hf_tk, files_metadata=True)

print("safetensors attr:", getattr(info, "safetensors", None))
print("\nSiblings sample:")
for s in info.siblings:
    print(f"  {s.rfilename} -> size: {s.size}")