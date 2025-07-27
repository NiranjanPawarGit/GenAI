import gradio as gr
from transformers import pipeline

# Load the GPT-2 model
story_gen = pipeline("text-generation", model="pranavpsv/gpt2-genre-story-generator")

def generate_story(prompt):
    result = story_gen(prompt, max_length=200, do_sample=True)
    return result[0]["generated_text"]

description = "Story generation with GPT-2"
title = "Generate your own story"
examples = [["Adventurer is approached by a mysterious stranger in the tavern for a new quest."]]

interface = gr.Interface(
    fn=generate_story,
    inputs=gr.Textbox(lines=3, placeholder="Enter a story prompt here..."),
    outputs="text",
    title=title,
    description=description,
    examples=examples
)

interface.launch()
