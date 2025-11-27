# 📘 English-to-French Seq2Seq Translator (TensorFlow/Keras — No Attention)

A complete, beginner-friendly **Encoder-Decoder LSTM translation model** that converts English sentences to French **without using any Attention mechanism**.  
This project demonstrates every step of the Seq2Seq NLP pipeline — preprocessing, vocabulary building, training with teacher forcing, greedy decoding, and saving model components.  
The entire implementation is fully commented for clarity.

---

## ⭐ Features

- ✔️ Clean **Seq2Seq Encoder–Decoder** architecture  
- ✔️ **No Attention** — pure LSTM hidden-state transfer  
- ✔️ Custom **tokenization, vocab building, and padding**  
- ✔️ **Teacher forcing** for training  
- ✔️ **Greedy decoding** for inference  
- ✔️ Saves encoder, decoder & full translation model  
- ✔️ Perfect for NLP beginners learning Seq2Seq  

---

## 🧠 Model Architecture

### 1. Encoder
- Tokenized English input  
- Embedding layer  
- LSTM → final hidden (`h`) and cell (`c`) states  
- These states initialize the decoder  

### 2. Decoder
- Embedding layer  
- LSTM → outputs sequence  
- Dense softmax to choose next French word  

### 3. Inference
Greedy decoding loop:
1. Feed `<start>`  
2. Predict next token  
3. Feed prediction back in  
4. Stop on `<end>`  

---

## 🔧 Installation

pip install tensorflow keras numpy pandas nltk scikit-learn

### Download NLTK tokenizer:

import nltk
nltk.download('punkt')

## 🎯 Why No Attention?

Training without Attention helps beginners understand:

How encoder final states carry sentence meaning

How decoder depends on fixed context vectors

Why long sentences are hard for classic Seq2Seq

Builds foundation before learning Attention or Transformers

## 🏗️ Extend the Project

You can add:

Attention mechanism (Luong/Bahdanau)

Beam search decoding

Subword tokenization (SentencePiece/BPE)

Larger datasets (Tatoeba, OpenSubtitles)

## 📄 License

MIT License

## 🤝 Contributions

Feel free to open:

Issues

Pull requests

Suggestions
