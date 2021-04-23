import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, Speech2TextFeatureExtractor,  Speech2TextTokenizer
import numpy as np

def translate(data, sampling_rate, pretrained_model_name="facebook/s2t-small-librispeech-asr"):
    model = Speech2TextForConditionalGeneration.from_pretrained(pretrained_model_name)
    
    feature_extractor = Speech2TextFeatureExtractor.from_pretrained(pretrained_model_name)
    tokenizer = Speech2TextTokenizer.from_pretrained(pretrained_model_name)
    processor = Speech2TextProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    inputs = processor(data, sampling_rate=sampling_rate, return_tensors="pt")
    generated_ids = model.generate(input_ids=inputs["input_features"], attention_mask=inputs["attention_mask"])

    transcription = processor.batch_decode(generated_ids)
    
    return transcription
    

    