import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

stopwords_nltk = stopwords.words('danish')
stemmer_nltk = SnowballStemmer("danish")


def lemmatize(text):
  lemmatizer = WordNetLemmatizer()
  tokens = word_tokenize(text)

  sentence_lemmatized = ""

  for token in tokens:
    token_lemmatized = lemmatizer.lemmatize(token)    
    sentence_lemmatized = sentence_lemmatized + " " + token_lemmatized
  
  return sentence_lemmatized.strip().lower()


def stem(text):
  stemmer = PorterStemmer()
  tokens = word_tokenize(text)

  sentence_stemmed = ""

  for token in tokens:
    token_stemmed = stemmer.stem(token)
    sentence_stemmed = sentence_stemmed + " " + token_stemmed
  
  return sentence_stemmed.strip().lower()


def remove_stopwords(text):
  stop_words = set(stopwords.words('english')) 

  word_tokens = word_tokenize(text)

  filtered_sentence = [w for w in word_tokens if not w in stop_words]
  sentence_no_stopwords = " ".join(filtered_sentence)
  
  return sentence_no_stopwords.strip()


def string_to_chunks(string_input, n):
    string_split = string_input.split(" ")
    
    sentences_chunks = []
    
    # looping till length l
    for i in range(0, len(string_split), n): 
        words_chunk =  string_split[i:i + n]
        sentence = " ".join(words_chunk)
        sentences_chunks.append(sentence)
        
    return sentences_chunks
  

def tokenize_and_stem(text):
    tokens = [token for token in word_tokenize(text, language="danish")]

    #exclude stopwords from stemmed words
    stems = [stemmer_nltk.stem(t) for t in tokens if t not in stopwords_nltk]

    return stems
  
  
  
# -- Other useful functions ---

test_string = "this string contains the word gaucho"
relevant_strings = ["gaucho", "space", "pool"]
# Check if string contains any of the words in list:
if any(word in test_string.lower() for word in relevant_strings):
    print("String contains word in relevant strings array")
