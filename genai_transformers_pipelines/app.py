import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import json

generator = pipeline("text-generation", model="gpt2", device=0)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)
model_name= "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name).to("cuda")


def run_pipeline(task, text):
    if task == "Text Generation":
        output = generator(text, max_length=50, num_return_sequences=1, truncation=True)
        result = output[0]['generated_text']
        return {"Generated Text": result} 
        
    
    elif task == "Summarization":
        output = summarizer(text, max_length=30, min_length=10, do_sample=False)
        summary_text = output[0].get("summary_text", "No summary generated")
        return {"Summary": str(summary_text)}
    
    elif task == "Embeddings":
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = embedding_model(**inputs)
            
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings_list = embeddings[0].cpu().numpy().tolist()
        # json_embedding = json.dumps(embeddings_list)
        return {"Embeddings": embeddings_list}
    
    return "Unkown Task"


gr.Interface(
    fn= run_pipeline,
    inputs= [
        gr.Dropdown(["Text Generation", "Summarization", "Embeddings"], label="Select Task"),
        gr.Textbox(lines=8, label="Enter Text"),
    ],
    outputs = gr.JSON(),
    title="GenAI Pipeline Playrround",
    description= "Try out Text Generation, Summarization, and Embedding extraction using Hugging Face Transformers."
).launch()      
