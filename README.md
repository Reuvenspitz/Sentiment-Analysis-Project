#  Sentiment Analysis of Movie Reviews 

This project was my gateway into NLP. I took a hands-on approach to compare different sentiment analysis techniques, from classical baselines to modern transformers, to gain a deeper understanding of their trade-offs.

I progressed through three main stages:
1.  **Classical ML:** Built baselines using TF-IDF with Logistic Regression and SVM.
2.  **RNNs:** Implemented a Bi-LSTM and ran a systematic study on different word embeddings (Random, GloVe, custom Word2Vec), **analyzing the impact of fine-tuning versus using frozen embeddings.**
3.  **Transformers:** Fine-tuned a RoBERTa model to benchmark against a state-of-the-art architecture.
   
All models were tested on both long-form **IMDB** reviews and short **Rotten Tomatoes** snippets.


##  Tech Stack

*   **Core:** Python, PyTorch, Scikit-learn, Pandas, NumPy
*   **NLP:** Transformers (Hugging Face), Gensim, NLTK/spaCy (for tokenization)
*   **Tools:** Jupyter Notebook, Git, GitHub


The following tables provide a detailed breakdown of the model performance on each dataset.

**Reproducibility & Experimental Setup:**
- All models were trained and evaluated on identical data splits
- All random processes (splitting, initialization, etc.) use a fixed seed defined in `src/config.py`
- Hyperparameters were selected through iterative experimentation rather than exhaustive grid search. The goal was to demonstrate the relative strengths of different approaches (classical vs. deep learning, different embedding strategies) rather than squeeze out maximum performance from each model.


---

### **IMDB Dataset**

#### Classical ML Model Results

| Model              	| Vectorizer      	| Vocabulary Size 	| Key Params                                    	| Test Accuracy 	|
|--------------------	|-----------------	|-----------------	|-----------------------------------------------	|---------------	|
| LogisticRegression 	| CountVectorizer 	| 20000           	| C=0.1,ngram=(1, 1)                            	| 87.26%        	|
| LogisticRegression 	| TfidfVectorizer 	| 20000           	| C=1,ngram_range=(1, 1)                        	| 87.93%        	|
| LogisticRegression 	| TfidfVectorizer 	| 20000           	| C=1,ngram_range=(1, 2)                        	| 88.15%        	|
| LogisticRegression 	| TfidfVectorizer 	| 19536           	| C=1,ngram_range=(1, 2), min_df=10, max_df=0.5 	| 88.60%        	|
| LinearSVC          	| TfidfVectorizer 	| 20000           	| C=0.1,ngram_range=(1, 2), min_df=10, max_df=5 	| 88.72%        	|


#### Deep Learning Model Results

| Method  	| Embedding Strategy    	| Key Hyperparams 	| Epochs Run 	| Time (s) 	| Best Val Acc 	| Final Test Acc 	|
|---------	|-----------------------	|-----------------	|------------	|----------	|--------------	|----------------	|
| Bi-LSTM 	| Random (Frozen)       	| embed_dim=200   	| 13         	| 794      	| 89.56%       	| 89.45%         	|
| Bi-LSTM 	| Random (Fine-Tuned)   	| embed_dim=200   	| 11         	| 701      	| 90.90%       	| 90.07%         	|
| Bi-LSTM 	| GloVe (Frozen)        	| embed_dim=300   	| 11         	| 676      	| 91.00%       	| 91.07%         	|
| Bi-LSTM 	| GloVe (Fine-Tuned)    	| embed_dim=300   	| 6          	| 347      	| 91.58%       	| 91.01%         	|
| Bi-LSTM 	| Word2Vec (Frozen)     	| embed_dim=256   	| 9          	| 508      	| 91.38%       	| 91.29%         	|
| Bi-LSTM 	| Word2Vec (Fine-Tuned) 	| embed_dim=256   	| 6          	| 346      	| 90.44%       	| 90.14%         	|
| RoBERTa 	| Fine-Tuned            	| lr=1e-6         	| 8          	| 5614     	| 94.20%       	| 94.50%         	|
<br>

### **Rotten Tomatoes Dataset**

#### Classical ML Model Results

| Model              	| Vectorizer      	| Vocabulary Size 	| Key Params                                   	| Test Accuracy 	|
|--------------------	|-----------------	|-----------------	|----------------------------------------------	|---------------	|
| LogisticRegression 	| CountVectorizer 	| 3000            	| C=0.1,ngram=(1, 1)                           	| 73.16%        	|
| LogisticRegression 	| TfidfVectorizer 	| 3000            	| C=1,ngram_range=(1, 1)                       	| 73.69%        	|
| LogisticRegression 	| TfidfVectorizer 	| 3000            	| C=1,ngram_range=(1, 2)                       	| 74.20%        	|
| LogisticRegression 	| TfidfVectorizer 	| 3000            	| C=1,ngram_range=(1, 2), min_df=5, max_df=0.5 	| 74.64%        	|
| LinearSVC          	| TfidfVectorizer 	| 3000            	| C=1,ngram_range=(1, 2), min_df=5, max_df=0.5 	| 74.61%        	|

#### Deep Learning Model Results

| Method  	| Embedding Strategy    	| Key Hyperparams 	| Epochs Run 	| Time (s) 	| Best Val Acc 	| Final Test Acc 	|
|---------	|-----------------------	|-----------------	|------------	|----------	|--------------	|----------------	|
| Bi-LSTM 	| Random (Frozen)       	| embed_dim=125   	| 13         	| 240      	| 77.36%       	| 75.58%         	|
| Bi-LSTM 	| Random (Fine-Tuned)   	| embed_dim=125   	| 9          	| 161      	| 79.26%       	| 76.29%         	|
| Bi-LSTM 	| GloVe (Frozen)        	| embed_dim=300   	| 13         	| 274      	| 80.48%       	| 78.90%         	|
| Bi-LSTM 	| GloVe (Fine-Tuned)    	| embed_dim=300   	| 5          	| 71       	| 81.18%       	| 78.06%         	|
| Bi-LSTM 	| Word2Vec (Frozen)     	| embed_dim=256   	| 14         	| 228      	| 82.20%       	| 80.64%         	|
| Bi-LSTM 	| Word2Vec (Fine-Tuned) 	| embed_dim=256   	| 5          	| 88       	| 81.32%       	| 78.23%         	|
| RoBERTa 	| Fine-Tuned            	| lr=1e-6         	| 10         	| 473      	| 89.98%       	| 87.26%         	|

*Note: Unless specified otherwise in `Key Hyperparams`, all Bi-LSTM models used a consistent architecture (`num_layers=2`, `pooling=attention`) and training setup (Optimizer: `AdamW`, Initial LR: `1e-3`, Scheduler: `ReduceLROnPlateau`, Loss: `CrossEntropy` with label smoothing).*

##  Key Insights & Analysis

1.  **Transformers Leverage Pre-trained Knowledge in Low-Context Scenarios.** RoBERTa's massive **+9.8%** accuracy gain on the short-form Rotten Tomatoes dataset stems from its vast pre-trained knowledge. It uses this knowledge to interpret nuance, sarcasm, and complex relationships from minimal text, whereas the LSTM is limited to the patterns found in the smaller training set. The gain on the text-rich IMDB dataset was a much smaller **+2.5%**, as the LSTM could extract sufficient context from the longer reviews.

2.  **Classical Models are a Fast and Efficient Baseline.** On the IMDB dataset, a simple Logistic Regression with optimized TF-IDF features reached **88.7%** accuracy. These models are exceptionally fast, **training in seconds without requiring a GPU**, and offer highly efficient inference. This makes them a powerful and competitive baseline, especially for text-rich classification tasks where nuanced context is less critical.
   
3.  **Frozen Pre-Trained Embeddings Often Outperform Fine-Tuning.** For the LSTM models, using *frozen* pre-trained embeddings (both GloVe and the custom Word2Vec) almost always yielded better or more stable results than fine-tuning. This suggests that freezing prevents the model from overwriting strong, general-purpose semantic features.

4.  **Corpus-Specific Embeddings Provide a Significant Edge.** The custom Word2Vec model, trained on the entire text corpus (including the RT and unsupervised data), gave the LSTM its best performance on the IMDB dataset. This highlights the value of domain-specific pre-training, even with a relatively small custom corpus.
---

## Project Structure

The repository is organized to separate preprocessing, experiments, and reusable code for clarity and maintainability.
```
├── notebooks/
│   ├── 00_Data_Preprocessing.ipynb          # Cleans, splits, and saves all datasets
│   ├── 01_Classical_ML_Baselines.ipynb      # Experiments with TF-IDF, SVM, Logistic Regression
│   ├── 02_Train_Word2Vec_Embeddings.ipynb   # Script to train custom embeddings on the corpus
│   ├── 03_LSTM_Experiments.ipynb            # All Bi-LSTM experiments with different embeddings
│   └── 04_RoBERTa_Finetuning.ipynb          # Fine-tuning the RoBERTa transformer model
├── src/
│   ├── __init__.py                          # Makes 'src' a Python package
│   ├── config.py                            # Central configuration for all paths and constants
│   └── trainer.py                           # Reusable training and evaluation functions for PyTorch models
├── .gitignore                               # Specifies files and directories to be ignored by Git
├── LICENSE                                  # MIT License
├── README.md                                # You are here!
└── requirements.txt                         # Project dependencies for easy setup
```

---

## Getting Started

To set up and run this project locally, please follow these steps.

### 1. Clone the Repository
```bash
git clone https://github.com/Reuvenspitz/Sentiment-Analysis-Project.git
cd Sentiment-Analysis-Project
```

### 2. Set Up the Environment

It is recommended to use a Python virtual environment.
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt
```

### 3. Data Setup

The datasets are not included in this repository. As defined in `src/config.py`, the project expects a parent directory containing both the project and data folders.

#### Create the data folders:
```
/your/chosen/workspace/
├── Sentiment-Analysis-Project/  (this repository)
└── Sentiment_Analysis_Data/
    └── raw/
```

Place the raw `.csv` files inside the `raw/` subdirectory.

### 4. Run the Pipeline

The notebooks are designed to be run in numerical order:

1. **(Optional)** Run `notebooks/02_Train_Word2Vec_Embeddings.ipynb` to generate the custom Word2Vec embeddings.
2. **(Required)** Run `notebooks/00_Data_Preprocessing.ipynb` to clean the raw data and create the necessary train/validation splits.
3. **(Experiments)** Once the clean data is generated, you can run the experiment notebooks (01, 03, 04) to reproduce the results.
---

## Acknowledgments

This project was conducted as part of a summer internship at the **DEEProsody Lab** at the **Weizmann Institute of Science**, headed by **Prof. David Harel** and led by **Dr. Tirza Biron**.

Special thanks to **Yaron Winter** for his guidance and helpful feedback throughout the project, and to the **Weizmann Institute** for providing access to its computational resources, which greatly supported the experiments.
