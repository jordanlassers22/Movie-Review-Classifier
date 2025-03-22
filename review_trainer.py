import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from review_prepper import ReviewPrepper
import numpy as np


def train_model():
    #Check if Keras is using a GPU (via TensorFlow)
    gpu_count = len(tf.config.list_physical_devices('GPU'))
    if gpu_count > 0:
        print("GPUs Available: {gpu_count}")
        print(tf.config.list_physical_devices('GPU'))
    else:
        print("No GPUs available, reverting to CPU")
    
    prepper = ReviewPrepper() #Instantiate custom review object
    
    #Get training data
    tokenized_training_reviews, y_train = prepper.load_and_tokenize_reviews("train", shuffle_results=True) #Load and preprocess training data
    top_10000_words = prepper.get_top_words(tokenized_training_reviews, 10000) #Extract the 10,000 most common words to use as features
    X_train = prepper.prepare_data_for_model(top_10000_words, tokenized_training_reviews) #Convert tokenized training reviews to bag-of-words binary features
    
    #Get testing data
    tokenized_testing_reviews, y_test = prepper.load_and_tokenize_reviews("test", shuffle_results=True) #Load and preprocess testing data
    X_test = prepper.prepare_data_for_model(top_10000_words, tokenized_testing_reviews) #Convert tokenized testing reviews using the same top words
    
    #Convert data to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    #Create Keras Model
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)), #Input layer with feature vector size
        keras.layers.Dense(128, activation='relu'), #First hidden layer
        keras.layers.Dropout(0.3), #Dropout to prevent overfitting
        keras.layers.Dense(64, activation='relu'), #Second hidden layer
        keras.layers.Dense(1, activation='sigmoid')  #Binary classification
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    #Early stopping to stop training when validation loss stops improving
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    #Train the model on the training data, validate on the test data
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1)
    
    #Show final model performance
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Validation Accuracy: {accuracy:.4f}")
    return model, top_10000_words
