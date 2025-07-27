from transformers import pipeline


summarizer = pipeline("summarization", device=0, model="sshleifer/distilbart-cnn-12-6")
text = """The Hugging Face Transformers library provides pre-trained models for a wide range\
    of natural language processing (NLP) tasks."""
result = summarizer(text, max_length=30, min_length=10, do_sample=False)
print("*************************************************\n")
print(result[0]['summary_text'])
