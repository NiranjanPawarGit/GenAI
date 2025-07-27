from transformers import pipeline

generator = pipeline("text-generation", model="gpt2", device=0)
prompt = "Once upon atime in a worldof AI"
result = generator(prompt, max_length=500, num_return_sequences=1)
print(result[0]["generated_text"])