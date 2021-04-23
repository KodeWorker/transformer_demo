import numpy as np
from scipy.io.wavfile import read
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, Speech2TextFeatureExtractor,  Speech2TextTokenizer

if __name__ == "__main__":
        
    audio_path = "../data/review#1.wav"
    samplerate, data = read(audio_path)
    
    data = (data - np.mean(data))/np.std(data)
    print(samplerate, len(data))
    
    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")

    feature_extractor = Speech2TextFeatureExtractor.from_pretrained("facebook/s2t-small-librispeech-asr")
    tokenizer = Speech2TextTokenizer.from_pretrained("facebook/s2t-small-librispeech-asr")
    processor = Speech2TextProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    inputs = processor(data, sampling_rate=samplerate, return_tensors="pt")
    generated_ids = model.generate(input_ids=inputs["input_features"], attention_mask=inputs["attention_mask"])

    transcription = processor.batch_decode(generated_ids)
    
    print(transcription)