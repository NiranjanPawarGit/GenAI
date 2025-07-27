from transformers import AutoTokenizer, AutoModel
import torch

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

sentence = "Transformers are revolutionizing NLP"
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to("cuda")
with torch.no_grad():
    outputs = model(**inputs)
    
#mean pooling
embeddings = outputs.last_hidden_state.mean(dim=1)
print("Embeddings vector shape:", embeddings.shape)



