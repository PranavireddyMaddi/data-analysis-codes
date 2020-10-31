# Movie Recommendation based on overview of movie 
# Content based Recommender System 

import os
os.chdir("D:\\CQ\\Python\\datasets")

import pandas as pd
movie_data = pd.read_csv("movies_metadata.csv",low_memory = False)
movie_data.shape
movie_data.columns
movie_data.overview


from sklearn.feature_extraction.text import TfidfVectorizer

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words="english")

# replacing the NaN values in overview column with
# empty string
movie_data["overview"].isnull().sum()
movie_data["overview"] = movie_data["overview"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming
# data
# Selecting only 20000 records - for time being 
movie_data_sampl = movie_data.iloc[:20000,:]

tfidf_matrix = tfidf.fit_transform(movie_data_sampl.overview)
tfidf_matrix.shape # 45466 x 75827

# with the above matrix we need to find the 
# similarity score
# There are several metrics for this
# such as the euclidean, the Pearson and 
# the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity 
# between 2 movies 
# Cosine similarity - metric is independent of 
# magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)


# creating a mapping of movie title to index number 
movie_index = pd.Series(movie_data_sampl.index, index=movie_data_sampl['title']).drop_duplicates()

movie_index["All's Faire in Love"]

def get_movie_recommendations(movie_title,topN):
    
    #movie_title="All's Faire in Love"
    #topN = 10
    # Getting the movie index using its title 
    movie_id = movie_index[movie_title]
    
    # Getting the pair wise similarity score for all the movies with that 
    # movie
    cosine_scores = list(enumerate(cosine_sim_matrix[movie_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar movies 
    cosine_scores_10 = cosine_scores[1:topN+1]
    
    # Getting the movie index 
    movie_idx  =  [i[0] for i in cosine_scores_10]
    movie_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar movies and scores
    movie_similar_show = pd.DataFrame(columns=["Title","Score"])
    movie_similar_show["Title"] = movie_data_sampl.loc[movie_idx,"title"]
    movie_similar_show["Score"] = movie_scores
    movie_similar_show.reset_index(inplace=True)  
    movie_similar_show.drop(["index"],axis=1,inplace=True)
    print (movie_similar_show)
    #return (movie_similar_show)

    
# Enter your movie and number of movies to be recommended 
get_movie_recommendations("Toy Story",topN=15)
