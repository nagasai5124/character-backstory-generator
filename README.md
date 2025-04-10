# character-backstory-generator
# DistilBERT
DistilBERT is pretrained by knowledge distillation to create a smaller model with faster inference and requires less compute to train. Through a triple loss objective during pretraining, language modeling loss, distillation loss, cosine-distance loss, DistilBERT demonstrates similar performance to a larger transformer language model.

# Model: DistilGPT-2
Base Model: distilgpt2 (a distilled version of OpenAI’s GPT-2)

Advantages:

Smaller and faster than GPT-2

Lower resource consumption

Still capable of creative, coherent text generation

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
