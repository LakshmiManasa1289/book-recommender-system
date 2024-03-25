import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data loading
final = pd.read_csv(r'C:\Users\study\Downloads\csvfiles\output_file.csv')
recommendations = final.sort_values(by='Book_Rating', ascending=False).head(5)

# Convert 'User_ID' and 'Book_Rating' to numeric types if they are not
final['User_ID'] = pd.to_numeric(final['User_ID'], errors='coerce')
final['Book_Rating'] = pd.to_numeric(final['Book_Rating'], errors='coerce')

# Drop NaN values if necessary
final = final.dropna(subset=['User_ID', 'Book_Rating'])

# Create the pivot table
pt = final.pivot_table(index='Book_Title', columns='User_ID', values='Book_Rating', fill_value=0)
pt.fillna(0, inplace=True)

def recommend(Book_name, use_min_user_ratings, min_user_ratings, use_min_similarity_score, min_similarity_score, use_min_total_ratings, min_total_ratings):
    index = np.where(pt.index == Book_name)[0][0] 
    similarity_scores = cosine_similarity(pt)
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
    
    # Filter based on minimum user ratings
    if use_min_user_ratings:
        similar_items = [(i, score) for i, score in similar_items if pt.iloc[i].values.sum() >= min_user_ratings]
    
    # Filter based on minimum similarity score
    if use_min_similarity_score:
        similar_items = [(i, score) for i, score in similar_items if score >= min_similarity_score]
    
    # Filter based on minimum total ratings
    if use_min_total_ratings:
        similar_items = [(i, score) for i, score in similar_items if pt.iloc[i].values.sum() >= min_total_ratings]
    
    recommended_books = [pt.index[i[0]] for i in similar_items]
    return recommended_books

def recommend_by_rating(rating, use_min_user_ratings, min_user_ratings):
    # Filter books with the specified rating and minimum user ratings
    filtered_books = final[(final['Book_Rating'] == rating) & (final.groupby('Book_Title')['User_ID'].transform('count') >= min_user_ratings)]
    
    # Get unique book titles
    recommended_books = filtered_books['Book_Title'].unique()[:5]
    
    return recommended_books

def generate_book_list_by_rating(rating):
    # Filter books with the specified rating
    books_with_rating = final[final['Book_Rating'] == rating]['Book_Title'].unique()
    
    return books_with_rating

def main():
    st.title("Book Recommendation System")
    st.header("User-Based Recommender's System")

    book_name = st.text_input("Enter book name or Book Title Here")  # User-based

    # Enable/Disable options for minimum user ratings
    use_min_user_ratings = st.checkbox("Enable Minimum User Ratings")
    if use_min_user_ratings:
        min_user_ratings = st.slider("Minimum User Ratings", min_value=1, max_value=10, value=5)
    else:
        min_user_ratings = None

    # Enable/Disable options for minimum similarity score
    use_min_similarity_score = st.checkbox("Enable Minimum Similarity Score")
    if use_min_similarity_score:
        min_similarity_score = st.slider("Minimum Similarity Score", min_value=0.1, max_value=1.0, step=0.1, value=0.5)
    else:
        min_similarity_score = None

    # Enable/Disable options for minimum total ratings
    use_min_total_ratings = st.checkbox("Enable Minimum Total Ratings")
    if use_min_total_ratings:
        min_total_ratings = st.slider("Minimum Total Ratings", min_value=0, max_value=500, value=0)
    else:
        min_total_ratings = None

    if st.button("Recommend"):
        try:
            book_name = int(book_name)
            if book_name in pt.columns:
                user_ratings = pt[book_name].dropna()
                top_rating_books = user_ratings.sort_values(ascending=False).head(10)

                st.subheader(f"Top 5 rated books for User {book_name}:")
                st.write(top_rating_books)
            else:
                st.warning("Invalid user id")
        except ValueError:
            if book_name in pt.index:
                recommended_books = recommend(book_name, use_min_user_ratings, min_user_ratings, use_min_similarity_score, min_similarity_score, use_min_total_ratings, min_total_ratings)
                st.subheader(f"Books similar to '{book_name}':")
                st.write(recommended_books)
                
                # Display total number of ratings in the sidebar
                st.sidebar.subheader(f"Total Ratings for '{book_name}':")
                st.sidebar.write(pt.loc[book_name].sum())
            else:
                st.warning(f"Not found")

    st.sidebar.title("User-Based")
    book_input = st.text_input("Enter book name")

    if st.button("Recommend similar books"):
        if book_input in pt.index:
            recommended_books = recommend(book_input, use_min_user_ratings, min_user_ratings, use_min_similarity_score, min_similarity_score, use_min_total_ratings, min_total_ratings)
            st.subheader("Similar books are:")
            st.write(recommended_books)
            
            # Display total number of ratings in the sidebar
            st.sidebar.subheader(f"Total Ratings for '{book_input}':")
            st.sidebar.write(pt.loc[book_input].sum())
        else:
            st.warning("Not found")

    # New button for recommending books with a specific rating
    rating_input = st.number_input("Enter a specific rating to get book recommendations", min_value=1, max_value=10, value=5, step=1)
    if st.button("Recommend books with specific rating"):
        recommended_books_by_rating = recommend_by_rating(rating_input, use_min_user_ratings, min_user_ratings)
        st.subheader(f"Books with rating {rating_input}:")
        st.write(recommended_books_by_rating)
        
        # New feature: Generate a list of books with the specified rating
        st.subheader(f"Books with rating {rating_input} in the dataset:")
        books_with_rating = generate_book_list_by_rating(rating_input)
        st.write(books_with_rating)

if __name__ == "__main__":
    main()
