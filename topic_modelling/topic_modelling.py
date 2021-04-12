import spacy
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

sentence_model = SentenceTransformer("bert-base-uncased", device="cuda")
sentence_model.max_seq_length = 500
model = KeyBERT(model=sentence_model)
nlp = spacy.load("en_core_web_sm")


document_input = "some interesting document here"
doc = nlp(document_input)

extracted_keywords = []

# Sentence disambiguation: 
for sent in doc.sents:
  keywords = model.extract_keywords(str(sent), keyphrase_ngram_range=(2, 2), stop_words='english', use_mmr=True, diversity=0.6,top_n=5)
  keywords_out = [extracted_keywords.append(word[0]) for word in keywords]

print("Extracted keywords")
print(extracted_keywords)
