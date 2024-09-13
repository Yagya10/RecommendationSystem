import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from main_code import *
from API_Invocations import *

# Function to display recommended papers in Streamlit
def display_recommendations(ranked_papers):
    """
    Displays the recommended papers with their scores in collapsible tabs.
    """
    if ranked_papers:
        for i, (paper, score) in enumerate(ranked_papers, 1):
            with st.expander(f"Paper {i}: {paper['title']} (Score: {score:.2f})"):
                st.markdown(f"**Title**: {paper['title']}")
                st.markdown(f"**Authors**: {', '.join(paper['author'])}")
                st.markdown(f"**Link**: [Read Here]({paper['link']})")
                st.markdown(f"**Published Date**: {paper['published_date']}")
                st.markdown("---")
    else:
        st.warning("No recommendations found.")

# Function to display the analytics page with graphical results and evaluation metrics
def display_analytics(paper_titles, content_scores, collaborative_scores, combined_scores, true_labels, predicted_labels):
    y_pos = np.arange(len(paper_titles))
    
    plt.figure(figsize=(12, 6))
    plt.barh(y_pos, content_scores, align='center', alpha=0.5, label='Content-Based')
    plt.barh(y_pos, collaborative_scores, align='center', alpha=0.5, label='Collaborative Filtering')
    plt.barh(y_pos, combined_scores, align='center', alpha=0.5, label='Hybrid')
    
    plt.yticks(y_pos, paper_titles)
    plt.xlabel('Recommendation Scores')
    plt.title('Comparison of Recommendation Scores')
    plt.gca().invert_yaxis()
    plt.legend()

    # Save the plot as a PNG file and display it in Streamlit
    plt.savefig('recommendation_scores.png')  # Save the plot to a file
    st.image('recommendation_scores.png')  # Display the saved image in Streamlit
    """
    Visualizes the recommendation scores and displays the evaluation metrics.
    """
    # Visualize the recommendation scores
    st.subheader("Comparison of Recommendation Scores")
    visualize_scores(paper_titles, content_scores, collaborative_scores, combined_scores)
    
    # Evaluation metrics (precision, recall, F1-score)
    st.subheader("Evaluation Metrics")
    precision, recall, f1 = evaluate_model(true_labels, predicted_labels)
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

# Main Streamlit App
def main():
    # Set page configuration
    st.set_page_config(page_title="Scientific Literature Recommendation System", layout="wide")
    
    # Title
    st.title("Recommendation System for Scientific Literature")
    
    # Sidebar Navigation
    pages = ["Home/Recommendations", "Analytics"]
    page = st.sidebar.selectbox("Navigate", pages)
    
    # Home/Recommendations Page
    if page == "Home/Recommendations":
        st.header("Get Paper Recommendations")
        search_query = st.text_input("Enter Search Query", placeholder="e.g., Machine Learning")
        user_id = st.number_input("Enter User ID (for collaborative filtering)", min_value=0, value=0)
        max_results = st.slider("Number of Results to Fetch", min_value=1, max_value=5, value=5)
        
        if st.button("Get Recommendations"):
            if search_query:
                # Fetch papers from APIs and generate recommendations
                papers = fetch_all_papers(search_query, max_results)
                if papers:
                    # Perform hybrid recommendations
                    interaction_matrix = np.array([
                        [1, 0, 3, 0, 5],  # Sample user-item interaction matrix
                        [0, 2, 0, 0, 1],
                        [4, 0, 0, 1, 0],
                        [0, 5, 0, 0, 0],
                        [2, 1, 0, 4, 0]
                    ])

                    ranked_papers, content_scores, collaborative_scores, combined_scores = hybrid_recommendations(
                        user_id, search_query, papers, interaction_matrix
                    )
                    
                    st.subheader(f"Top {min(max_results, len(ranked_papers))} Recommended Papers")
                    display_recommendations(ranked_papers[:max_results])
                else:
                    st.warning("No papers found for the given query.")
            else:
                st.warning("Please enter a search query.")
    
    # Analytics Page
    elif page == "Analytics":
        st.header("Analytics: Graphical Visualizations and Evaluation Metrics")
        
        # Example placeholder data (use real data in practice)
        paper_titles = ["Paper 1", "Paper 2", "Paper 3", "Paper 4", "Paper 5"]
        content_scores = [0.6, 0.7, 0.4, 0.8, 0.5]
        collaborative_scores = [0.5, 0.6, 0.3, 0.7, 0.4]
        combined_scores = [0.55, 0.65, 0.35, 0.75, 0.45]
        true_labels = [1, 1, 0, 1, 0]  # Example ground truth labels
        predicted_labels = [1, 1, 0, 1, 0]  # Example predicted labels

        display_analytics(paper_titles, content_scores, collaborative_scores, combined_scores, true_labels, predicted_labels)

if __name__ == "__main__":
    main()