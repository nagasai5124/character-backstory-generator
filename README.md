# character-backstory-generator
Character Background Story Generator is an AI-powered tool that creates rich and imaginative backstories for fictional characters using the lightweight DistilGPT-2 language model. Designed for writers, game developers, and RPG enthusiasts, this tool helps bring characters to life with compelling and unique narratives.

# DistilBERT
DistilBERT is pretrained by knowledge distillation to create a smaller model with faster inference and requires less compute to train. Through a triple loss objective during pretraining, language modeling loss, distillation loss, cosine-distance loss, DistilBERT demonstrates similar performance to a larger transformer language model.

# Model: DistilGPT-2
Base Model: distilgpt2 (a distilled version of OpenAI’s GPT-2)

Advantages:

Smaller and faster than GPT-2

Lower resource consumption

Still capable of creative, coherent text generation

# Architecture


Frontend (e.g., Streamlit or Flask)
   |
   |-- User Input: Character name, Race,class
   |
Backend (Python)
   |
   |-- Prompt Engineering: Constructs input prompt for DistilGPT-2
   |
   |-- Text Generation: Generates story using Hugging Face's Transformers
   |
   |-- Output: Sends story back to frontend for display


# Dependencies
transformers
torch
streamlit


# how to run and install 
To create a Python environment (using venv), open your terminal, navigate to your project directory, and run 

1, python3 -m venv

2, pip install -r /path/to/requirements.txt

# Running Streamlit code in Google Colab involves a few steps.

Streamlit is a Python library that is typically used to create web applications for data science and machine learning projects.

# ! pip install streamlit -q
The second line (!wget -q -O - ipv4.icanhazip.com) retrieves your external IP address using the wget command.

# wget -q -O - ipv4.icanhazip.com

Copy IP address

The line %%writefile app.py writes the Streamlit app code to a file named app.py.
# !streamlit run app.py & npx localtunnel --port 8501
to expose the locally running Streamlit app to the internet. The app is hosted on port 8501

# outputs
![output image](https://github.com/nagasai5124/character-backstory-generator/blob/main/Streamlit%20-%20Google%20Chrome%204_9_2025%206_18_26%20PM.png)



https://github.com/user-attachments/assets/56a84ee8-833c-4f0b-aa4f-8e748c2362ef


