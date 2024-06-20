import streamlit as st
import soundfile as sf
import numpy as np
import tempfile
import os
import logging
import pyaudio
import wave
import threading
import wave
from datetime import datetime
from audio_recorder_streamlit import audio_recorder


from voice_agent.stt_model import STTModel
from voice_agent.tts_model import TTSModel
from voice_agent.llm_model import LLMModel
from voice_agent.save_wav import save_wav_file


logging.basicConfig(filename='logs/model_logs.txt', level=logging.INFO, format='%(asctime)s %(message)s')


stt_model = STTModel()
tts_model = TTSModel()
llm_model = LLMModel()

st.title("Voice Agent with Local LLM and Models")

st.subheader("Base audio recorder")
base_audio_bytes = audio_recorder(key="base")
if base_audio_bytes:
    st.audio(base_audio_bytes, format="audio/wav")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file_path = f'inputs/recording_{timestamp}.wav'
    with open(input_file_path, 'wb') as f:
            f.write(base_audio_bytes)



    logging.info(f"App: Saved recording to {input_file_path}")
    logging.info("App: Processing recording.")
    
    # Convert audio to text
    transcription = stt_model.transcribe(input_file_path)
    st.write(f"Transcription: {transcription}")

    # Process text with LLM
    response = llm_model.generate_response(transcription)
    st.write(f"LLM Response: {response}")

    # Convert text response to speech using TTS
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_speaker:
        speaker_wav = tmp_speaker.name
        audio_input, _ = sf.read(input_file_path)
        sf.write(speaker_wav, audio_input, 16000)  # Use the recorded audio as the speaker's voice

    tts_audio_path = tts_model.synthesize(response)

    # Play TTS audio in Streamlit
    st.audio(tts_audio_path, format='audio/wav')

    # Clean up
    os.remove(speaker_wav)
    os.remove(tts_audio_path)

    logging.info("App: Completed processing and playback.")
else:
    st.write("Please record some audio and then click 'Process Recording'")