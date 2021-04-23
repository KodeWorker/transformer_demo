import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, Speech2TextFeatureExtractor,  Speech2TextTokenizer
from datasets import load_dataset
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np

if __name__ == "__main__":

    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    
    feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
    tokenizer = Speech2TextTokenizer.from_pretrained("facebook/s2t-small-librispeech-asr")
    processor = Speech2TextProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    def map_to_array(batch):
        speech, _ = sf.read(batch["file"])
        batch["speech"] = speech
        return batch

    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.map(map_to_array)

    data = ds["speech"][0]
    scaled = np.int16(data/np.max(np.abs(data)) * 32767)
    write('../data/test.wav', 16000, scaled)
    
    print(len(data))
    inputs = processor(data, sampling_rate=16_000, return_tensors="pt")
    generated_ids = model.generate(input_ids=inputs["input_features"], attention_mask=inputs["attention_mask"])

    transcription = processor.batch_decode(generated_ids)
    
    print(transcription)