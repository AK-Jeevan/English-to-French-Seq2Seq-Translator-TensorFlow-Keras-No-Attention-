# English -> French seq2seq (Encoder-Decoder) WITHOUT Attention.
# Fully commented (each line / method explained) for beginners.

import numpy as np                             # numerical arrays and operations
import pandas as pd                            # DataFrame I/O and manipulation
import tensorflow as tf                        # main deep learning framework
from tensorflow import keras                   # high-level Keras API (models, layers, training)
from keras.models import Model                 # functional Model class to build encoder-decoder
from keras.layers import Input, LSTM, Embedding, Dense  # core layers used in seq2seq
from keras.callbacks import EarlyStopping      # callback to stop training early
from keras.preprocessing.sequence import pad_sequences  # pads sequences to fixed length
from nltk.tokenize import word_tokenize        # tokenize sentences into word tokens (NLTK)
import nltk                                    # to download tokenizer resources

# Download NLTK punkt tokenizer data if not present (quiet=True suppresses messages)
nltk.download('punkt', quiet=True)

# -----------------------------
# 1. Load dataset
# -----------------------------
# CSV expected to have two columns: English sentence (col 0) and French sentence (col 1).
pairs = pd.read_csv(r"C:\Users\akjee\Documents\AI\NLP\NLP-DL\Encoder-Decoder\en-fr-without-Attention.csv")  # read csv into pandas DataFrame
print("Loaded pairs:", pairs.shape)            # print shape (rows, cols) for quick check

# Convert first two columns to Python lists of strings (ensures proper types)
eng_sentences = pairs.iloc[:, 0].astype(str).tolist()  # .iloc selects column by index; .astype(str) ensures strings
fr_sentences  = pairs.iloc[:, 1].astype(str).tolist()  # .tolist() converts Series to plain list

# -----------------------------
# 2. Tokenization and vocabulary creation
# -----------------------------
def build_vocab_and_tokenize(sentences):
    """
    Tokenize sentences and build word->index and index->word mappings.
    - sentences: list of strings
    Returns:
      tokenized: list[list[str]] token lists for each sentence
      word2idx: dict mapping token -> integer id (with special tokens)
      idx2word: reverse mapping
    """
    tokenized = []                               # will hold token lists
    vocab = set()                                # temporary set to collect unique tokens

    for s in sentences:
        toks = word_tokenize(s.lower())          # word_tokenize splits sentence into words; .lower() normalizes
        tokenized.append(toks)                   # append token list
        vocab.update(toks)                       # add tokens to vocab set

    # Reserve special tokens with fixed ids:
    # '<pad>' = 0 for padding positions
    # '<start>' = 1 used to indicate decoder start
    # '<end>' = 2 used to indicate decoder end
    word2idx = {'<pad>': 0, '<start>': 1, '<end>': 2}
    next_id = 3

    # Sort vocab for deterministic ordering, then assign indices starting from next_id
    for w in sorted(vocab):
        if w not in word2idx:
            word2idx[w] = next_id
            next_id += 1

    # Create reverse mapping idx->word for decoding predictions back to text
    idx2word = {i: w for w, i in word2idx.items()}

    return tokenized, word2idx, idx2word

# Build English and French tokenizations & vocabularies
eng_tok, eng2idx, idx2eng = build_vocab_and_tokenize(eng_sentences)  # returns token lists and maps for English
fr_tok,  fr2idx, idx2fr  = build_vocab_and_tokenize(fr_sentences)   # returns token lists and maps for French

print("English vocab size:", len(eng2idx))   # show how many tokens (incl. special tokens)
print("French vocab size:", len(fr2idx))

# -----------------------------
# 3. Convert tokens to index sequences and pad
# -----------------------------
def encode_and_pad(tokenized_sentences, word2idx, max_len=None):
    """
    Map tokens to integer ids, add <start> and <end>, then pad/truncate to max_len.
    - tokenized_sentences: list[list[str]]
    - word2idx: mapping token->id
    - max_len: if None, compute from data (longest sequence + 2 for start/end)
    Returns:
      padded_ids: numpy array shape (num_sentences, max_len) dtype=int32
    """
    sequences = []
    for toks in tokenized_sentences:
        # Map tokens to ids, using <pad> id (0) for unknown words (safe fallback)
        ids = [word2idx.get(t, word2idx['<pad>']) for t in toks]   # list comprehension maps each token -> id
        # Add decoder-style start and end tokens to the sequence
        ids = [word2idx['<start>']] + ids + [word2idx['<end>']]    # prepends start, appends end
        sequences.append(ids)                                      # collect sequence

    # If max_len not provided, choose the length of the longest sequence
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    # pad_sequences right-pads with the value 0 (default) to length max_len
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post', value=word2idx['<pad>'])
    return padded.astype(np.int32)

# Compute sequences and determine a fixed max_len (keeps model inputs uniform)
max_len = None
X = encode_and_pad(eng_tok, eng2idx, max_len=max_len)   # encoder input (English)
Y = encode_and_pad(fr_tok, fr2idx, max_len=X.shape[1])  # decoder input (French) - use same length as encoder for simplicity

print("Encoder/Decoder sequence shape:", X.shape, Y.shape)

# -----------------------------
# 4. Prepare decoder targets (shifted by one timestep)
# -----------------------------
# For teacher forcing we provide decoder input Y (starts with <start>) and teacher target is Y shifted left by 1.
# Create Y_target of same shape as Y and fill as: Y_target[:, t] = Y[:, t+1] for t < last, and 0 for last position.
Y_target = np.zeros_like(Y)              # initialize target array with zeros (pad id)
Y_target[:, :-1] = Y[:, 1:]              # shift left: target at t is decoder token at t+1

# -----------------------------
# 5. Train/Validation split
# -----------------------------
from sklearn.model_selection import train_test_split  # import here to keep code flow logical
# Split the dataset into train and validation sets; stratify not used (sequence tasks often not categorical per-sample)
X_train, X_val, Y_train, Y_val, Yt_train, Yt_val = train_test_split(
    X, Y, Y_target, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Build Encoder-Decoder model (no attention)
# -----------------------------
latent_dim = 256                         # size of LSTM hidden states (increase for better capacity)
embedding_dim = 128                      # word embedding size

# ---------- Encoder ----------
encoder_inputs = Input(shape=(X_train.shape[1],), dtype='int32', name='encoder_inputs')  # placeholder for encoder integer sequences
encoder_embedding_layer = Embedding(input_dim=len(eng2idx), output_dim=embedding_dim, mask_zero=True, name='encoder_embedding')
encoder_embedded = encoder_embedding_layer(encoder_inputs)     # maps token ids -> dense vectors (embedding lookup)

# LSTM encoder returns final hidden state and final cell state when return_state=True
encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')  # LSTM layer object
_, state_h, state_c = encoder_lstm(encoder_embedded)                     # run LSTM, discard outputs, keep states
encoder_states = [state_h, state_c]                                      # bundle states for passing to decoder

# ---------- Decoder ----------
decoder_inputs = Input(shape=(Y_train.shape[1],), dtype='int32', name='decoder_inputs')  # placeholder for decoder input sequences
decoder_embedding_layer = Embedding(input_dim=len(fr2idx), output_dim=embedding_dim, mask_zero=True, name='decoder_embedding')
decoder_embedded = decoder_embedding_layer(decoder_inputs)               # embedding lookup for decoder inputs

# LSTM decoder returns the full sequence outputs and final states; it is initialized with encoder states
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')  # return_sequences needed to predict at each timestep
decoder_outputs, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)            # run decoder LSTM

# Dense layer projects LSTM outputs to probability distribution over French vocabulary at each timestep
decoder_dense = Dense(len(fr2idx), activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)   # final output shape: (batch_size, timesteps, vocab_size)

# Instantiate the full training model that maps [encoder_input, decoder_input] -> decoder_output probabilities
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # functional API combines input/output tensors

# Compile the model:
# - optimizer='adam' is efficient adaptive optimizer
# - loss='sparse_categorical_crossentropy' expects integer class labels per timestep (not one-hot)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()   # prints model architecture summary (layers, shapes, params)

# -----------------------------
# 7. Training with EarlyStopping
# -----------------------------
# Use EarlyStopping to stop training when validation loss stops improving (prevents overfitting).
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Keras expects y_true shape either (batch, timesteps) for sparse loss; we pass Yt_train which is integer ids per timestep.
history = model.fit(
    [X_train, Y_train],                # inputs: encoder sequences and decoder sequences (teacher forcing)
    Yt_train,                          # targets: next-token indices (shifted decoder sequences)
    validation_data=([X_val, Y_val], Yt_val),  # validation inputs and targets
    batch_size=64,                     # number of samples per gradient update
    epochs=100,                        # maximum epochs to train
    callbacks=[early_stop],            # early stopping callback
    verbose=1
)

# -----------------------------
# 8. Build inference models for prediction (translate)
# -----------------------------
# Encoder inference model: given an input sequence, returns encoder states (h, c)
encoder_model = Model(encoder_inputs, encoder_states)  # maps encoder input -> final states

# Decoder inference model:
# - At inference we feed one token at a time and update states step-by-step.
# - We reuse the decoder embedding layer, the decoder LSTM layer, and the dense layer defined above.

# Inputs for the decoder inference: one-token input and previous states
decoder_state_input_h = Input(shape=(latent_dim,), name='dec_input_h')  # placeholder for hidden state
decoder_state_input_c = Input(shape=(latent_dim,), name='dec_input_c')  # placeholder for cell state
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Reuse the same embedding layer; feed one-timestep input (shape=(1,))
decoder_single_input = Input(shape=(1,), dtype='int32', name='decoder_single_input')
decoder_single_embedded = decoder_embedding_layer(decoder_single_input)      # maps single token id -> embedding

# Run the decoder LSTM for one time step using provided states
decoder_outputs2, state_h2, state_c2 = decoder_lstm(decoder_single_embedded, initial_state=decoder_states_inputs)

# Project LSTM output to vocabulary probabilities
decoder_outputs2 = decoder_dense(decoder_outputs2)   # shape: (batch=1, time=1, vocab_size)
decoder_states2 = [state_h2, state_c2]               # updated states to feed in next step

# Final decoder model for inference: input = (token, prev_h, prev_c) -> output_probs, next_h, next_c
decoder_model = Model(
    [decoder_single_input] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2
)

# -----------------------------
# 9. Translation helper (inference loop)
# -----------------------------
def decode_sequence(input_seq, max_len=Y.shape[1]):
    """
    Translate a single encoded English input sequence to French text.
    - input_seq: numpy array shape (1, timesteps) with encoder token ids
    - max_len: maximum number of decoder timesteps to generate
    Returns: translated sentence string
    """
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq, verbose=0)   # returns [h, c] for the input sequence

    # Initialize target sequence with the <start> token id
    target_seq = np.array([[fr2idx['<start>']]])                 # shape (1,1)

    stop_condition = False
    decoded_tokens = []                                          # list to collect output token strings

    for _ in range(max_len):
        # Predict next token probabilities and next states using decoder_model
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        # output_tokens shape: (1, 1, vocab_size) -> take the probabilities for the single timestep
        sampled_token_index = int(np.argmax(output_tokens[0, -1, :]))  # pick the token with highest prob (greedy)
        sampled_word = idx2fr.get(sampled_token_index, '<unk>')        # convert token id -> word (fallback '<unk>')

        # Stop if we reach the end token
        if sampled_word == '<end>' or sampled_word == '<pad>':
            break

        decoded_tokens.append(sampled_word)   # append predicted word to output list

        # Update the target sequence (the predicted token becomes the next input)
        target_seq = np.array([[sampled_token_index]])

        # Update states for next iteration
        states_value = [h, c]

    # Join tokens into a single string (space-separated)
    return ' '.join(decoded_tokens)

# -----------------------------
# 10. Test translation on a few examples
# -----------------------------
# Take a few samples from the validation set, encode them and run inference
for i in range(5):
    input_seq = X_val[i:i+1]                  # slice single example (shape (1, timesteps))
    print("Input (English tokens):", input_seq)
    translation = decode_sequence(input_seq)  # run decode_sequence to obtain predicted French
    print("Predicted French:", translation)
    print("Reference French tokens:", Y_val[i])  # printed as token ids (for quick check)
    print("-" * 50)

# -----------------------------
# 11. Save model components (optional)
# -----------------------------
model.save("en_fr_seq2seq_no_attention.h5")    # save full training model (includes encoder+decoder training graph)
encoder_model.save("en_fr_encoder.h5")         # save encoder inference model
decoder_model.save("en_fr_decoder.h5")         # save decoder inference model
print("Models saved: en_fr_seq2seq_no_attention.h5, en_fr_encoder.h5, en_fr_decoder.h5")