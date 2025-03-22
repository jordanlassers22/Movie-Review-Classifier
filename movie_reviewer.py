import os
from tensorflow import keras
from review_trainer import train_model
import tkinter as tk
from tkinter import filedialog, messagebox
from review_prepper import ReviewPrepper
import numpy as np

MODEL_PATH = "saved_model/my_model.keras"
TOP_WORDS_PATH = "saved_model/top_10000_words.txt"

#If the model file exists, load it. Otherwise, train and save it.
try:
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = keras.models.load_model(MODEL_PATH)
        
        #Load top words from saved file
        os.makedirs(os.path.dirname(TOP_WORDS_PATH), exist_ok=True)
        with open(TOP_WORDS_PATH, "r", encoding="utf-8") as f:
            top_words = [line.strip() for line in f.readlines()]
    else:
        print("No saved model detected...")
        print("Training new model...")
        model, top_words = train_model()
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)
        os.makedirs(os.path.dirname(TOP_WORDS_PATH), exist_ok=True)
        with open(TOP_WORDS_PATH, "w", encoding="utf-8") as f:
            for word in top_words:
                f.write(f"{word}\n")
        print(f"Model and top words saved to: {MODEL_PATH}, {TOP_WORDS_PATH}")
        
except Exception:
    print("No saved model detected...")
    print("Training new model...")
    model, top_words = train_model()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    os.makedirs(os.path.dirname(TOP_WORDS_PATH), exist_ok=True)
    with open(TOP_WORDS_PATH, "w", encoding="utf-8") as f:
        for word in top_words:
            f.write(f"{word}\n")
    print(f"Model and top words saved to: {MODEL_PATH}, {TOP_WORDS_PATH}")
    
#Create main window
root = tk.Tk()
root.title("Movie Review Classifier")
root.geometry("600x500")

#Text area where review can be written
text_area = tk.Text(root, height=15, width=70, wrap='word')
text_area.pack(pady=10)

#Add placeholder text
placeholder_text = "Write your review here..."
text_area.insert(tk.END, placeholder_text)

prepper = ReviewPrepper()
def classify_review():
    review = text_area.get("1.0", tk.END).strip()
    if not review or review == placeholder_text:
        messagebox.showwarning("Blank input", "Please write or paste a review.")
        return
    
    clean_review = prepper.clean_text(review)
    tokenized_review = prepper.tokenize_text(clean_review)
    X_input = prepper.prepare_data_for_model(top_words, [tokenized_review])
    X_input = np.array(X_input)
    prediction = model.predict(X_input)[0][0]
    
    review = ""
    if prediction >= 0.5:
        review = "Positive"
    else:
        review = "Negative"
    
    messagebox.showinfo("Prediction", f"This is a {review} review. ")

#Button to trigger classification
classify_button = tk.Button(root, text="Classify Review", command=classify_review)
classify_button.pack(pady=10)

#Drag and drop area for file
def load_text_file():
    filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if filepath:
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            text_area.delete("1.0", tk.END)
            text_area.insert(tk.END, content)

drop_area = tk.Label(root, text="📂 Click here to open a text review.", relief="ridge", borderwidth=2, width=60, height=4, bg="lightgray")
drop_area.pack(pady=15)
drop_area.bind("<Button-1>", lambda e: load_text_file())

root.mainloop()