import streamlit as st
import re
import numpy as np
import pickle
from keras.models import load_model

#Load the LSTM Model
model=load_model('model.h5')

with open('vocab.pkl','rb') as handle:
    vocab=pickle.load(handle)
    
# Create a mapping from character to unique index..
char_to_int = {u:i for i, u in enumerate(vocab)}

# Create a mapping from indices to characters. This is
#   the inverse of char_to_ind and allows us to convert back
#   from unique index to the character in our vocabulary.
int_to_char = np.array(vocab)

# function to convert the characters to numeric format
def vectorize(lyrics):
    """function to vectorizing the text into its numeric representation"""
    vect_lyrics = np.array([char_to_int[i] for i in lyrics])
    return vect_lyrics

# function for preprocessing the text
def preprocessing(text):
    """function to perform preprocessing on the text"""
    # removing the nextline characters
    text = text.replace("\\n"," ")
    # making the text consistent by replacing all the characters by their uncapitalized characters
    text = text.lower()
    # removing all the remaining brackets
    text = re.sub(r'\[.*?\]'," ",text)
    # removing all special characters except numberd and alphabets
    text = re.sub("[^a-z0-9'\.\n]"," ", text)
    # removing all the unnecessary spaces
    text = re.sub(' +', ' ', text)
    # removing the leading or trailing spaces from the text
    text = text.strip()
    # returning the output
    return text

# function to predict song lyrics
def generate_chars(input, length):
  """Function used for predicting the song lyrics based on the input"""
  try:
    # defining the sequence length
    seq_length = 50
    output = []
    # making sure that the input text is fully pre-processed
    input = preprocessing(input)
    # defining scenarios for inputs of different lengths
    # model can take input only of sequence length as it has been trained on that sequence length
    # input length > sequence length
    if len(input) > seq_length:
      # adding the input to the output list
      output.append(input)
      # getting the length from where the input should be taken
      n = len(input) - seq_length
      # defining the input of seq_length
      input = input[n:]
      # making sure that the input text is fully pre-processed
      ip = preprocessing(input)
    # input length < sequence length
    elif len(input) < seq_length:
       # adding the input to the output list
      output.append(input)
      # making sure that the input text is fully pre-processed
      ip = preprocessing(input)
       # defining the input of seq_length
      ip = ip.rjust(50," ")
    # input length == sequence length
    else:
       # adding the input to the output list
      output.append(input)
      # making sure that the input text is fully pre-processed
      ip = preprocessing(input)
    # converting the text into its numeric form
    ip = vectorize(ip)
    # reshaping the data into required format so that it can be given to the model
    x_ip = np.reshape(ip,(1,seq_length,1))
    # normalizing the input data
    x_ip = x_ip / len(vocab)
    # looping till we get the output till required character length
    for i in range(0,length,1):
      # getting the prediction which is in the form of probabilities
      y_prob = model.predict(x_ip)
      # getting the max value from the probabilities
      y_pred = y_prob.argmax(axis = 1)
      # appending the predicted character to the output list
      output.extend(int_to_char[y_pred])
      # normalizing the predicted text which is in its numeric form
      norm = y_pred / len(vocab)
      # adding the prediction to the input data and making sure that the input is of length 50
      x_ip = np.reshape((np.append(x_ip,norm)[1:]),(1,seq_length,1))
    # returning the output
    return output
  except:
    return "An error occured. Please try with another song"

# streamlit app
st.title("Lyrics Prediction With LSTM")
st.write("Note: Please keep in mind that the lyrics sometimes will not make any sense as the LSTM is trained on characters rather than words")
input_text=st.text_input("Enter the sequence of characters of atleast 50 characters","I call you when I need you, my heart's on fire You come to me, come to me")
lyrics_char_length = st.text_input("Enter the number of characters you want as output",50)
if st.button("Predict Lyrics"):
    if len(input_text) >= 50: # lyrics should be of atleast 50 character length
        output = generate_chars(input_text,int(lyrics_char_length))
        if "error" in output:
            st.error("An error occured. Please try with another song lyrics",icon="🚨")
        else:
            lyrics = ""
            for i in output:
                lyrics+=i
            st.write(f'Lyrics: {lyrics}')
    else:
        st.error('Please make sure that the entered lyrics has atleast 50 characters', icon="🚨")
        
