from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
from torch import bfloat16
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
torch.random.manual_seed(0)

logging.basicConfig(filename='logs/model_logs.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# class LLMModel:
#     def __init__(self):
#         self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#         self.model = GPT2LMHeadModel.from_pretrained("gpt2")

#     def generate_response(self, input_text):
#         logging.info(f"LLMModel: Generating response for input text - {input_text}")
#         inputs = self.tokenizer.encode(input_text, return_tensors="pt")
#         outputs = self.model.generate(inputs, max_length=50, num_return_sequences=1)
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logging.info(f"LLMModel: Generated response - {response}")
#         return response
    

class LLMModel:
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16  # Computation type
        )

        self.device_map = {
            "transformer.wte": 0,  
            "transformer.h": "cpu",  
            "lm_head": 0 
        }
                
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=self.quantization_config,
            device_map=self.device_map
        )
        self.model.eval()
        
        self.pipe = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer,
            task='text-generation',
            model_kwargs={"torch_dtype": bfloat16},
            temperature=0.1,
            max_new_tokens=500,
            repetition_penalty=1.1
        )
    
    def generate_response(self, input_text):
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant. Here is an input from user, please give him answer"},
            {"role": "user", "content": input_text},
        ]
        
        terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.pipe(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        
        response = outputs[0]["generated_text"][-1]['content']
        return response
