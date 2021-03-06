from danlp.models import load_bert_ner_model
import streamlit as st

st.write('# Danish NER')

@st.cache 
def predict_ner_tags(text):
    bert = load_bert_ner_model()
    # Get lists of tokens and labels in BIO format
    tokens, labels = bert.predict(text)
    print(" ".join(["{}/{}".format(tok,lbl) for tok,lbl in zip(tokens,labels)]))
    return labels


labels = predict_ner_tags("Andreas Mogensen er den første danske astronaut")
st.write(labels)