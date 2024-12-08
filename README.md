# 🎵 Lyrics Generation with LSTM  

This project demonstrates the use of a character-based **LSTM (Long Short-Term Memory)** model for generating song lyrics. The trained model generates lyrics based on a sequence of characters provided by the user. A **Streamlit** web application is included to make the model interactive and user-friendly.  

## 🚀 Features  
- **Character-Based Lyrics Generation:** Predicts lyrics one character at a time based on the given input sequence.  
- **Preprocessing Pipeline:** Cleans and normalizes input text to ensure consistent predictions.  
- **Custom Output Length:** Allows users to specify the desired length of generated lyrics.  
- **Streamlit Interface:** Interactive web app for inputting lyrics and viewing the generated output.  

## 🛠️ Technologies Used  
- **Python**  
- **TensorFlow/Keras:** For building and training the LSTM model.  
- **Streamlit:** For the web-based user interface.  
- **Pandas, NumPy:** For data manipulation.  
- **Regex:** For text preprocessing.  

## 📂 Project Structure  
.
├── model.h5                # Trained LSTM model  
├── vocab.pkl               # Pickled vocabulary file  
├── app.py                  # Main Streamlit application  
├── experiments.ipynb       # Jupyter Notebook detailing the step-by-step process of building the project  
├── requirements.txt        # Python dependencies  
└── README.md               # Project documentation  

```  

## 🎯 How to Use  
1. **Clone the Repository:**  
   ```bash  
   git clone https://github.com/your-username/lyrics-generation-lstm.git  
   cd lyrics-generation-lstm  
   ```  

2. **Install Dependencies:**  
   Make sure you have Python installed. Install the required packages:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run the Streamlit App:**  
   Start the app locally using:  
   ```bash  
   streamlit run app.py  
   ```  
   The application will open in your browser.  

4. **Generate Lyrics:**  
   - Enter a sequence of at least 50 characters in the input box.  
   - Specify the desired output length (number of characters).  
   - Click "Predict Lyrics" to generate the lyrics.  

## 📖 Example  
### Input:  
```
"I call you when I need you, my heart's on fire You come to me, come to me"
```

### Output:  
```
"I call you when I need you, my heart's on fire you come to me, come to me i'll never let you"
```

*Note: The generated lyrics might not make complete sense as the model is character-based.*  

## 📌 Limitations  
- The model is trained on characters rather than words, so the output may lack coherence.  
- Limited training data may lead to repetitive or nonsensical predictions.  

## 🌟 Future Enhancements  
- Train on a larger and more diverse lyrics dataset.  
- Switch to a word-based or transformer-based approach for better coherence.  
- Add support for fine-tuning using user-provided datasets.  

## 📄 License  
This project is licensed under the MIT License.  

## 🤝 Contributions  
Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.  

---

Happy Coding! 🎤  
