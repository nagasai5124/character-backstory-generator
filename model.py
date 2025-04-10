#Importing Necessary Libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

#Load and Explore the Dataset
from datasets import load_dataset

ds = load_dataset("MohamedRashad/dnd_characters_backstories")
data=ds['train'].to_pandas()
data.head(10)

#data analysis

text=data['text'].iloc[0]
text=text.split('\n')
text=text[2].split(':')

#geting unique character name and race
Character_Race=[]
Character_Class=[]
for i in range(len(data)):
  text=data['text'].iloc[i].split('\n')
  char_text=text[2].split(':')
  class_text=text[3].split(':')
  if char_text[1] not in Character_Race:
    Character_Race.append(char_text[1])
  if class_text[1] not in Character_Class:
    Character_Class.append(class_text[1])

print(Character_Class)  #' Rogue/fighter/wizard', Warlock, hints of bard'

print(Character_Race)   # Human(pretends he's a high elf)", Dark elf/human',Warforged (robot),'1/2 pig, 1/2 goblin but looks like a pig'

print(f"number of Character Race: {len(Character_Race)}, number of Character Class : {len(Character_Class)}")

#Select the Device for Model Training
device='cuda' if torch.cuda.is_available() else 'cpu'

#Load the Tokenizer and Pre-trained Model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
tokenizer.pad_token = tokenizer.eos_token

# The transformer
model = GPT2LMHeadModel.from_pretrained('distilgpt2').to(device)
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=32

print(model)

#Dataset Preparation and Custom Dataset Class Definition
class gptDataset(Dataset):
  def __init__(self,data,tokenizer,max_length):
    self.data=data
    self.tokenizer=tokenizer
    self.max_length=max_length

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    text=str(self.data['text'].iloc[idx])
    target=str(self.data['target'].iloc[idx])
    final_text=f"{text} | {target}"
    tokens=self.tokenizer(final_text,
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors='pt')
    return tokens
data_sample=gptDataset(data,tokenizer,256)
#Dataset into Training and Validation Sets

'''
Training Set (80%): Used to train the model by optimizing its parameters. Validation Set (20%): Used to evaluate the model’s performance after each epoch without updating parameters.

'''
train_size = int(0.8 * len(data_sample))
valid_size = len(data_sample) - train_size
train_data, valid_data = random_split(data_sample, [train_size, valid_size])

print(f"train data size: {len(train_data)}, valid data size: {len(valid_data)}")


#Create Data Loaders

train_loader=DataLoader(train_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
val_loader=DataLoader(valid_data,batch_size=VAL_BATCH_SIZE)


#Model params

# Set the number of epochs
num_epochs = 4
# Model params
# Training parameters
model_name = 'distilgpt2'
gpu = 0
loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
num_train_steps = int(len(train_data) / TRAIN_BATCH_SIZE * num_epochs)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
)
tokenizer.pad_token = tokenizer.eos_token
# Init a results dataframe
results = pd.DataFrame(columns=['epoch', 'transformer', 'batch_size', 'gpu',
                                'training_loss', 'validation_loss', 'epoch_duration_sec'])

#Training and Validation Loop

def train_val_step(model,loss,optimizer,epochs,train_loader,val_loader,device,results,batch_size):
  #the training loop
  for epoch in range(epochs):
    start_time=time.time() # Start the timer for the epoch

    #training
    model.train()
    epoch_training_loss=0
    train_iterator=tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs} ")
    for batch in train_iterator:
      optimizer.zero_grad()
      inputs= batch['input_ids'].squeeze(1).to(device)
      targets=inputs.clone()
      outputs=model(input_ids=inputs,
                    labels=targets)
      loss=outputs.loss
      loss.backward()
      optimizer.step()
      scheduler.step()
      train_iterator.set_postfix({'Training Loss': loss.item()})
      epoch_training_loss += loss.item()
    avg_epoch_training_loss = epoch_training_loss / len(train_iterator)


    #validation
    model.eval()
    epoch_validation_loss=0
    total_loss=0
    valid_iterator = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}")
    with torch.no_grad():
      for batch in valid_iterator:
        inputs=batch['input_ids'].squeeze(1).to(device)
        targets=inputs.clone()
        outputs=model(input_ids=inputs,labels=targets)
        loss=outputs.loss
        total_loss+=loss.item()
        valid_iterator.set_postfix({'Validation Loss': loss.item()})
        epoch_validation_loss += loss.item()

    avg_epoch_validation_loss = epoch_validation_loss / len(val_loader)

    end_time = time.time()  # End the timer for the epoch
    epoch_duration_sec = end_time - start_time  # Calculate the duration in seconds

    new_row = {'transformer': model_name,
               'batch_size': batch_size,
               'gpu': gpu,
               'epoch': epoch+1,
               'training_loss': avg_epoch_training_loss,
               'validation_loss': avg_epoch_validation_loss,
               'epoch_duration_sec': epoch_duration_sec}  # Add epoch_duration to the dataframe

    results.loc[len(results)] = new_row
    print(f"Epoch: {epoch+1}, Validation Loss: {total_loss/len(val_loader)}")

  return results


final_result = train_val_step(model, loss, optimizer, num_epochs, train_loader, val_loader, device, results, batch_size=TRAIN_BATCH_SIZE)

print(final_result)

#Model Testing and Response Validation

input_str='Generate Backstory based on following information\nCharacter Name: Erryt\nCharacter Race: Aarakocra\nCharacter Class: Ranger\n\nOutput:\n'
encoded_input = tokenizer.encode_plus(
    input_str,
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=128  # Adjust max_length as needed
)

input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)

# Set the pad_token_id to the tokenizer's eos_token_id
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
    pad_token_id=pad_token_id
)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

path="distilgpt_3.pt"
model.save_pretrained(path)