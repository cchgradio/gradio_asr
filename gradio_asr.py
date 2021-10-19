import gradio as gr
import librosa
import soundfile as sf
import torch
import warnings

from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer

#load wav2vec2 tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

# define speech-to-text function
def asr_transcript(audio_file):
    transcript = ""

    # Stream over 20 seconds chunks
    stream = librosa.stream(
        audio_file.name, block_length=20, frame_length=16000, hop_length=16000
    )

    for speech in stream:
        if len(speech.shape) > 1:
            speech = speech[:, 0] + speech[:, 1]

        input_values = tokenizer(speech, return_tensors="pt").input_values
        logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)[0]
        transcript += transcription.lower() + " "

    return transcript

gradio_ui = gr.Interface(
    fn=asr_transcript,
    title="Speech-to-Text with HuggingFace+Wav2Vec2",
    description="Upload an audio clip, and let AI do the hard work of transcribing",
    inputs=gr.inputs.Audio(label="Upload Audio File", type="file"),
    outputs=gr.outputs.Textbox(label="Auto-Transcript"),
)

#gradio_ui.launch(share=True)
gradio_ui.launch()    
