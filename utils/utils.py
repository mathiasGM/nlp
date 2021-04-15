def string_to_chunks(string_input, n):
    string_split = string_input.split(" ")
    
    sentences_chunks = []
    
    # looping till length l
    for i in range(0, len(string_split), n): 
        words_chunk =  string_split[i:i + n]
        sentence = " ".join(words_chunk)
        sentences_chunks.append(sentence)
        
    return sentences_chunks
