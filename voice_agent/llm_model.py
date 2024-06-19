from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

logging.basicConfig(filename='logs/model_logs.txt', level=logging.INFO, format='%(asctime)s %(message)s')

class LLMModel:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def generate_response(self, input_text):
        logging.info(f"LLMModel: Generating response for input text - {input_text}")
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=50, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"LLMModel: Generated response - {response}")
        return response
