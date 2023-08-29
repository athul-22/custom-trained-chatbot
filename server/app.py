from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import os
from pathlib import Path

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    try:
        documents = SimpleDirectoryReader(directory_path).load_data()
    except Exception as e:
        print(f"Error loading data from directory: {e}")
        return None

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    try:
        index.save_to_disk('index.json')
    except Exception as e:
        print(f"Error saving index: {e}")

    return index

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="CUSTOM TRAINED AI")

# Specify the directory path where your training documents are located
directory_path = "/Users/athulnambiar/Desktop/ACHU/AI BOT/docs"

# Construct the index
index = construct_index(directory_path)

# Launch the Gradio interface
if index is not None:
    iface.launch(share=True)
else:
    print("Gradio interface not launched due to errors.")