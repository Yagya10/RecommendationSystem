import re
import nltk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.metrics import precision_score, recall_score, f1_score
from concurrent.futures import ThreadPoolExecutor

from API_Invocations import *

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to clean and preprocess text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to tokenize text and remove stopwords
def tokenize_text(text):
    """
    Tokenizes the input text and removes common stopwords using NLTK.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    
    return tokens

# Function to vectorize text using TF-IDF
def vectorize_text(texts):
    """
    Converts a collection of text documents to a matrix of TF-IDF features.
    - TF-IDF (Term Frequency-Inverse Document Frequency) gives importance to rare words in a document.
    """
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text data
    tfidf_matrix = vectorizer.fit_transform(texts)

    return tfidf_matrix, vectorizer

# Function to calculate cosine similarity between user query and document vectors
def calculate_cosine_similarity(query, documents):
    """
    Calculates cosine similarity between a query and a set of documents.
    - Cosine similarity measures the cosine of the angle between two vectors, which shows how similar they are.
    """
    # Combine query with documents for vectorization
    documents.insert(0, query)

    # Vectorize the text (query + documents)
    tfidf_matrix, vectorizer = vectorize_text(documents)

    # Vectorize the query (since it's the first item now)
    query_vector = tfidf_matrix[0:1]

    # Calculate cosine similarity between query and all documents
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[1:]).flatten()

    return cosine_similarities

# Function to perform collaborative filtering using SVD
def collaborative_filtering(interaction_matrix, user_id):
    """
    Performs matrix factorization using Singular Value Decomposition (SVD) 
    to predict user-item interactions (recommendations).
    """
    if not np.issubdtype(interaction_matrix.dtype, np.float64):
        interaction_matrix = interaction_matrix.astype(float)
    # Perform matrix factorization using SVD
    # U: user feature matrix, sigma: diagonal matrix of singular values, Vt: item feature matrix
    U, sigma, Vt = svds(interaction_matrix, k=2)

    # Convert sigma to a diagonal matrix
    sigma = np.diag(sigma)

    # Predict the user-item interaction matrix by multiplying the three matrices together
    predicted_matrix = np.dot(np.dot(U, sigma), Vt)

    # Get the predicted interactions for the specified user
    user_predictions = predicted_matrix[user_id, :]

    return user_predictions

# Function to combine Content-Based and Collaborative Filtering to create Hybrid Recommendations
def hybrid_recommendations(user_id, query, papers, interaction_matrix):
    """
    Generates hybrid recommendations by combining content-based filtering (cosine similarity)
    and collaborative filtering (SVD) results.
    """
    # Step 1: Clean and vectorize the abstracts for content-based filtering
    cleaned_abstracts = [clean_text(paper['abstract']) for paper in papers]
    tokenized_abstracts = [' '.join(tokenize_text(abstract)) for abstract in cleaned_abstracts]
    
    # Vectorize the tokenized abstracts (TF-IDF)
    tfidf_matrix, vectorizer = vectorize_text(tokenized_abstracts)
    
    # Cosine similarity between query and documents (Content-based Filtering)
    query_cleaned = clean_text(query)
    cosine_similarities = calculate_cosine_similarity(query_cleaned, cleaned_abstracts)

    # Step 2: Collaborative Filtering Predictions for User-Paper Interactions
    collaborative_predictions = collaborative_filtering(interaction_matrix, user_id)

    # Step 3: Ensure both content_similarities and user_predictions have the same length
    min_length = min(len(cosine_similarities), len(collaborative_predictions))
    cosine_similarities = cosine_similarities[:min_length]  # Truncate to minimum length
    collaborative_predictions = collaborative_predictions[:min_length]  # Truncate to minimum length

    # Step 4: Combine Content-Based and Collaborative Filtering Scores
    combined_scores = 0.5 * cosine_similarities + 0.5 * collaborative_predictions

    # Step 5: Rank papers by combined score
    ranked_papers = sorted(list(zip(papers[:min_length], combined_scores)), key=lambda x: x[1], reverse=True)
    
    return ranked_papers, cosine_similarities, collaborative_predictions, combined_scores

def visualize_scores(paper_titles, content_scores, collaborative_scores, combined_scores):
    """
    Visualizes the recommendation scores from content-based filtering, collaborative filtering, and hybrid methods.
    Generates a horizontal bar plot for easy comparison.
    """
    y_pos = np.arange(len(paper_titles))
    
    # Create the figure and plot three types of scores for comparison
    plt.figure(figsize=(12, 6))
    
    # Bar for content-based scores
    plt.barh(y_pos, content_scores, align='center', alpha=0.5, label='Content-Based')
    
    # Bar for collaborative filtering scores
    plt.barh(y_pos, collaborative_scores, align='center', alpha=0.5, label='Collaborative Filtering')
    
    # Bar for hybrid scores (combined)
    plt.barh(y_pos, combined_scores, align='center', alpha=0.5, label='Hybrid')
    
    # Setting labels, title, and other visual elements
    plt.yticks(y_pos, paper_titles)
    plt.xlabel('Recommendation Scores')
    plt.title('Comparison of Recommendation Scores (Content-Based, Collaborative, Hybrid)')
    plt.gca().invert_yaxis()  # Invert the y-axis to display the highest score on top
    plt.legend()  # Show a legend for the bar types
    plt.tight_layout()  # Adjust the layout to fit everything neatly
    
    # Show the plot
    plt.show()
# Function to evaluate the model using precision, recall, and F1-score
def evaluate_model(true_labels, predicted_labels):
    """
    Evaluates the performance of the recommendation model using:
    - Precision: The percentage of relevant items among the recommended ones.
    - Recall: The percentage of relevant items that were recommended.
    - F1-Score: The harmonic mean of precision and recall.
    """
    # Calculate precision
    precision = precision_score(true_labels, predicted_labels, average='weighted')

    # Calculate recall
    recall = recall_score(true_labels, predicted_labels, average='weighted')

    # Calculate F1-score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    # Print out the evaluation metrics
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    return precision, recall, f1

# Main Program to Fetch Data, Apply Filtering, and Display Results
if __name__ == "__main__":
    # Take search query and user ID as input
    search_query = input("Enter the search query: ")
    user_id = int(input("Enter user ID (for collaborative filtering): "))
    max_results = int(input("Enter the number of results to fetch from each source: "))

    # Step 1: Fetch papers from multiple APIs
    papers = fetch_all_papers(search_query, max_results)

    if papers:
        # Step 2: Perform hybrid recommendations (content-based + collaborative filtering)
        interaction_matrix = np.array([
            [1, 0, 3, 0, 5],  # Sample interaction data for user-item interactions
            [0, 2, 0, 0, 1],
            [4, 0, 0, 1, 0],
            [0, 5, 0, 0, 0],
            [2, 1, 0, 4, 0]
        ])  # Rows: Users, Columns: Papers

        # Step 3: Generate hybrid recommendations
        ranked_papers, content_scores, collaborative_scores, combined_scores = hybrid_recommendations(
            user_id, search_query, papers, interaction_matrix
        )

        # Step 4: Display top-ranked papers
        print(f"Top {min(10, len(ranked_papers))} Recommended Papers:")
        for i, (paper, score) in enumerate(ranked_papers[:10], 1):
            print(f"Rank {i}: {paper['title']} (Score: {score:.2f})")
            print(f"Authors: {', '.join(paper['author'])}")
            print(f"Link: {paper['link']}")
            print(f"Published Date: {paper['published_date']}")
            print("-" * 80)

        # Step 5: Visualization - Display recommendation scores
        paper_titles = [paper['title'] for paper, _ in ranked_papers[:10]]
        visualize_scores(paper_titles, content_scores[:10], collaborative_scores[:10], combined_scores[:10])

        # Step 6: Evaluation Metrics (mock true_labels for demonstration)
        # Assume some mock true labels (actual user feedback)
        true_labels = np.random.randint(2, size=10)  # Example: actual user engagement (0 or 1)
        predicted_labels = np.where(np.array(combined_scores[:10]) > 0.5, 1, 0)  # Predicted labels based on scores

        # Evaluate model performance
        evaluate_model(true_labels, predicted_labels)
    else:
        print("No papers found from the available APIs.")