import torch
from transformers import AutoTokenizer, AutoModelForPreTraining, MarianMTModel, MarianTokenizer


class Translator:
    
    def __init__(self):
        self.tokenizer_translator_en_da = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-da")
        self.model_translator_en_da = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-da")
        
        self.tokenizer_translator_da_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-da-en")
        self.model_translator_da_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-da-en")

    def translate_en_to_da(self, text_english):
        src_text = [text_english]
        translated = self.model_translator_en_da.generate(**self.tokenizer_translator_en_da.prepare_seq2seq_batch(src_text, return_tensors="pt"))
        output_text = [self.tokenizer_translator_en_da.decode(t, skip_special_tokens=True) for t in translated]
        return output_text[0]
    
    def translate_da_to_en(self, text_danish):
        src_text = [text_danish]
        translated = self.model_translator_da_en.generate(**self.tokenizer_translator_en_da.prepare_seq2seq_batch(src_text, return_tensors="pt"))
        output_text = [self.tokenizer_translator_da_en.decode(t, skip_special_tokens=True) for t in translated]
        return output_text[0]