import os
import re #Regex library
import random

#Modify load and tokenize reviews to return labels based off filename
class ReviewPrepper:
    
    def __init__(self):
        self.negation_words = [
            "no", "not", "neither", "nor", "none", 
            "nobody", "nothing", "nowhere", "never", 
            "barely", "hardly", "scarcely", "seldom", "rarely", "cant", "cannot"
        ]
        
        self.stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
        'ourselves', 'you', "youre", "youve", "youll", "youd", 'your', 'yours', 'yourself', 
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "shes", 'her', 'hers', 'herself', 
        'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
         'what', 'which', 'who', 'whom', 'this', 'that', "thatll", 'these', 'those', 'am', 'is', 'are',
         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
         'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
        'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
        's', 't', 'can', 'will', 'just', 'don', "dont", 'should', "shouldve", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
         'ain', 'aren', "arent", 'couldn', "couldnt", 'didn', "didnt", 'doesn', "doesnt", 
        'hadn', "hadnt", 'hasn', "hasnt", 'haven', "havent", 'isn', "isnt", 'ma', 'mightn',
         "mightnt", 'mustn', "mustnt", 'needn', 'neednt', 'shan', 'shant', 'shouldn', 'shouldnt', 
        'wasn', "wasnt", 'weren', "werent", 'won', "wont", 'wouldn', "wouldnt"]
    
    def clean_text(self, text):
        """ Cleans raw text by removing HTML tags and non-alphabetic characters."""
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)  #Remove HTML
        text = re.sub(r"[^a-zA-Z\s]", "", text)  #Remove non-alphabetic chracters
        words = text.split()
        filtered_words = []
        
        for word in words:
            if word not in self.stop_words:
                filtered_words.append(word)
        return " ".join(filtered_words)
    
    def tokenize_text(self, text):
        """ Tokenizes text into individual words."""
        words = text.split()  #Split words every space
        i = 0
        while i < len(words) - 1:
            word = words[i] #Get current word
            if word in self.negation_words: #If word negates the following word
                next_word = words.pop(i+1) #Get and remove the following word
                words[i] = f"{word}_{next_word}" #Combine both words with an underscore
            i+= 1
            
        return words
    
    def load_and_tokenize_reviews(self, base_directory, shuffle_results = False):
        """
        Loads movie reviews from a directory, cleans and tokenizes them.
        Returns a tuple (2d list of tokenized reviews, corresponding review label)
        Optional Parameter:
            shuffle_results: returns a shuffled 2d array of tokens to prevent bias in model training.
        """
        tokenized_reviews = []
        labels = []
    
        #Loop through 'pos' and 'neg' subdirectories
        for subdirectory in ["pos", "neg"]:
            dir_path = os.path.join(base_directory, subdirectory)  #Construct path to subdirectory
            if not os.path.exists(dir_path):
                continue  #Skip if the subdirectory does not exist
                
            print(f"Iterating through {dir_path}")
            
            #Loop through files in directory
            for filename in os.listdir(dir_path):
                if filename.endswith(".txt"):
                    with open(os.path.join(dir_path, filename), "r", encoding="utf-8") as f:
                        raw_text = f.read()
                        cleaned_text = self.clean_text(raw_text)
                        tokenized_text = self.tokenize_text(cleaned_text)
                        
                        #Extract label separately
                        label = self.extract_label_from_filename(filename)
    
                        #Append tokenized review and corresponding label
                        tokenized_reviews.append(tokenized_text)
                        labels.append(label)
    
        #Shuffle reviews and labels together if needed
        if shuffle_results and tokenized_reviews:
            combined = list(zip(tokenized_reviews, labels))  #Pair reviews and labels
            random.shuffle(combined)  #Shuffle pairs
            tokenized_reviews, labels = zip(*combined)  #Unzip into separate lists
    
            #Convert back to lists (since zip() returns tuples)
            tokenized_reviews = list(tokenized_reviews)
            labels = list(labels)
    
        return tokenized_reviews, labels
    
    def get_top_words(self, token_lists, n = 10000):
        """Takes in a 2d list of tokens. Returns the most common n-number of words. n=10000 by default"""
        word_counts = {}
        for token_list in token_lists:
            for token in token_list:
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1
                    
        sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True) #Turn word_counts to list and sort
        top_words = []
        for word, _ in sorted_words[:n]:
            top_words.append(word)
        
        return top_words
    
    def extract_label_from_filename(self, filename):
        """ Extracts label from filename based on rating (0-5 = Negative (0), 6-10 = Positive (1)). """
        score = int(filename.split("_")[1].split(".")[0])  #Extracts score from filename
        return 0 if score <= 5 else 1  
    
    def prepare_data_for_model(self, top_words, tokenized_reviews):
        """
        Converts tokenized reviews into a binary Bag-of-Words representation as a DataFrame.
        Parameters:
            top_words: List of top words to use as features.
            tokenized_reviews: 2D list of tokenized reviews.
        Returns:
            DataFrame where columns are top_words and rows are binary feature vectors.
        """
        # Initialize an empty list to collect feature vectors
        feature_vectors = []
    
        # Create word-to-index mapping
        word_index = {}
        for i, word in enumerate(top_words):
            word_index[word] = i
    
        # Process each review
        for review in tokenized_reviews:
            # Initialize feature vector for the current review
            feature_vector = [0] * len(top_words)  # Binary vector of zeros
            
            # Mark presence of words in the review
            for token in review:
                if token in word_index:
                    feature_vector[word_index[token]] = 1
            
            # Add feature vector to the list
            feature_vectors.append(feature_vector)
        
        
        return feature_vectors
        
        
    
    
if __name__ == "__main__":
    prepper = ReviewPrepper()
    tokenized_texts, labels = prepper.load_and_tokenize_reviews("train", shuffle_results=True)
    top_words = prepper.get_top_words(tokenized_texts)
    
    X = prepper.prepare_data_for_model(top_words, tokenized_texts)


