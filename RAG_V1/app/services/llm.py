from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import asyncio

class LLMService:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    async def generate_answer(self, question, context, conversation):
        prompt = self._build_prompt(question, context, conversation)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate, prompt)

    def _generate(self, prompt):
        max_length = self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') else 512
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=128)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()

    def _build_prompt(self, question, context, conversation):
        return (
            "You are a helpful assistant. Use the provided context to answer the user's question as clearly and thoroughly as possible. "
            "If the context contains steps or instructions, summarize them.\n"
            f"Context: {context}\nQuestion: {question}\nAnswer:"
        )