import gradio as gr
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_question(question, system_prompt="You're a helpful assitant."):
    messages= [
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":question}
    ]
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=messages
    )
    return response["choices"][0]['message']['content']


demo = gr.Interface(
    fn=ask_question,
    inputs=[
        gr.Textbox(label="Your Question"),
        gr.Dropdown(
            choices=[
                "You are a helful assitant.",
                "Answer like Yoda",
                "Answer in one sentence only.",
                "Be sarcastic.",
                "Explain like I'm five."
            ],
            label="System Prompt"
        )
    ],
    outputs = gr.Textbox(label="Response"),
    title="Gradio Q&A App with OpenAI",
    description="Try asking a question and modify the system prompt to see changes!"
)

demo.launch()