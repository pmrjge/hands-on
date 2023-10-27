from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))

print(classifier(["I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"]))

classifier = pipeline("zero-shot-classification")
print(classifier("This is a course about the Transformers library",
           candidate_labels=["education", "politics", "business"],))

generator = pipeline("text-generation")
print(generator("In this course, we will teach you how to"))

generator = pipeline("text-generation", model="distilgpt2")
print(generator("In this course, we will teach you how to", max_length=30, num_return_sequences=2,))

unmasker = pipeline("fill-mask")
print(unmasker("This course will teach you all about <mask> models.", top_k=2))

ner = pipeline("ner", grouped_entities=True)
print(ner("My name is Sylvain and I work at Hugging Face in Brooklyn."))

question_answerer = pipeline("question-answering")
print(question_answerer(question="Where do I work?", context="My name is Sylvain and I work at Hugging Face in Brooklyn"))

# summarizer = pipeline("summarization")
# summarizer("""....""")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
print(translator("Ce cours est produit par Hugging Face."))

# Text Classification
# Zero shot classification
# Text generation
# Text completion
# Token classification
# Question answering
# summarization
# Translation