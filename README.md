# Movie Review Classifier (Sentiment Analysis)

This is a sentiment analysis tool for classifying movie reviews as **positive** or **negative** using a custom-trained neural network. The model is trained from raw IMDB reviews and includes a drag-and-drop GUI built with Tkinter to allow 
for easy testing of reviews.

---

## Features

- Trains a binary classification neural network using Keras and TensorFlow
- Parses raw IMDB review files (no CSVs or external datasets)
- Custom tokenizer with HTML stripping, stop word removal, and negation handling
- Bag-of-Words vectorization using top 10,000 most frequent words
- Saves and reloads trained model and word list
- GUI to paste a review, load from `.txt`, and classify in real time
- Outputs prediction with confidence score

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/movie-review-classifier.git
   cd movie-review-classifier
   ```
2. Install dependencies
   ```bash
   pip install tensorflow
   pip install numpy
   pip install pandas
   pip install scikit-learn
   ```
## How to Use

### Train or Load Model
The app will automatically:
Load an existing model from saved_model / Or train a new one if no saved model exists

From the terminal enter:

```bash
python movie_reviewer.py
```
In the GUI:
Paste a movie review into the text box OR Click the gray area to load a .txt review file

Click "Classify Review" to predict the review's sentiment
## License
This project is licensed under the MIT License.
