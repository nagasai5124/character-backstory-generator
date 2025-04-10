
import streamlit as st
from transformers import AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
st.title("generate character backstory")
name=st.text_input("Character Name ")
race=st.text_input("Character Race ")
c_class =st.text_input("Character Class")
if st.button('generate'):
  text=f"Generate Backstory based on following information\nCharacter Name:{name}\nCharacter Race:{race}\nCharacter Class: {c_class}\n "
  model = AutoModelForCausalLM.from_pretrained("/content/distilgpt_3.pt")
  tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
  tokenizer.pad_token = tokenizer.eos_token

  encoded_input = tokenizer.encode_plus(
    text,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=128  # Adjust max_length as needed
  )

  input_ids = encoded_input['input_ids']
  attention_mask = encoded_input['attention_mask']
  pad_token_id = tokenizer.eos_token_id
  # Generate the output
  output = model.generate(
      input_ids,
      attention_mask=attention_mask,
      max_length=256,  # Adjust max_length as needed
      num_return_sequences=1,
      do_sample=True,
      top_k=8,
      top_p=0.95,
      temperature=1.5,
      repetition_penalty=1.2,
      max_new_tokens=300,
      pad_token_id=pad_token_id
  )

  decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
  st.write(decoded_output)
