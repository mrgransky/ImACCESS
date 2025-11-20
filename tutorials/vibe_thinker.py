from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class VibeThinker:
		def __init__(self, model_path):
				self.model_path = model_path
				self.model = AutoModelForCausalLM.from_pretrained(
						self.model_path,
						low_cpu_mem_usage=True,
						dtype="bfloat16",
						device_map="auto"
				)
				self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
				# Ensure a pad token exists (some chat models omit it)
				if self.tokenizer.pad_token is None:
					self.tokenizer.pad_token = self.tokenizer.eos_token
					self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
				
				if hasattr(self.tokenizer, "padding_side") and self.tokenizer.padding_side is not None:
					self.tokenizer.padding_side = "left"

		def infer_text(self, prompt):
				messages = [
						{"role": "user", "content": prompt}
				]
				text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
				model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

				text = self.tokenizer.apply_chat_template(
						messages,
						tokenize=False,
						add_generation_prompt=True
				)
				model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

				generation_config = dict(
						max_new_tokens=40960,
						do_sample=True,
						temperature=0.6, # 0.6 or 1.0, you can set it according to your needs
						top_p=0.95,
						top_k=None # in vLLM or SGlang, please set top_k to -1, it means skip top_k for sampling
				)
				generated_ids = self.model.generate(
						**model_inputs,
						generation_config=GenerationConfig(**generation_config)
				)
				generated_ids = [
						output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
				]

				response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

				return response


if __name__ == '__main__':
		model = VibeThinker('WeiboAI/VibeThinker-1.5B')
		prompt = 'What is the future of Bitcoin?'
		print(model.infer_text(prompt))