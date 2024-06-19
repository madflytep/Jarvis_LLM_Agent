import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging

logging.basicConfig(filename='logs/model_logs.txt', level=logging.INFO, format='%(asctime)s %(message)s')

class STTModel:
    def __init__(self, model_id="openai/whisper-medium"):
        logging.info(f"Loading Whisper model: {model_id}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, audio_file):
        logging.info("STTModel: Starting transcription.")
        result = self.pipe(audio_file)
        transcription = result["text"]
        logging.info(f"STTModel: Completed transcription - {transcription}")
        return transcription