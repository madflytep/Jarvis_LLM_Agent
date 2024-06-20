from TTS.api import TTS
import logging

logging.basicConfig(filename='logs/model_logs.txt', level=logging.INFO, format='%(asctime)s %(message)s')

class TTSModel:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True):
        self.tts = TTS(model_name, gpu=gpu)

    def synthesize(self, text, language="en"):
        logging.info(f"TTSModel: Starting synthesis for text - {text}")
        output_file = 'output.wav'
        self.tts.tts_to_file(text=text, file_path=output_file, speaker='Ana Florence',language=language)
        logging.info(f"TTSModel: Completed synthesis - saved to {output_file}")
        return output_file
