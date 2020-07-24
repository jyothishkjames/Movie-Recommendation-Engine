import numpy as np
import pandas as pd
import recommender_functions as rf
import sys


class Recommender:
    """
    This Recommender uses FunkSVD to make predictions of exact ratings.
    And uses either FunkSVD or a Knowledge Based recommendation (highest ranked)
    to make recommendations for users.  Finally, if given a movie, the recommender
    will provide movies that are most similar as a Content Based Recommender.
    """

    def __init__(self):
        pass

    def fit(self, reviews_pth, movies_pth, latent_features=12, learning_rate=0.0001, iters=100):
        """
        This function performs matrix factorization using a basic form of FunkSVD with no regularization

        INPUT:
        reviews_pth - path to csv with at least the four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies_pth - path to csv with each movie and movie information in each row
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations

        OUTPUT:
        None - stores the following as attributes:
        n_users - the number of users (int)
        n_movies - the number of movies (int)
        num_ratings - the number of ratings made (int)
        reviews - dataframe with four columns: 'user_id', 'movie_id', 'rating', 'timestamp'
        movies - dataframe of
        user_item_mat - (np array) a user by item numpy array with ratings and nans for values
        latent_features - (int) the number of latent features used
        learning_rate - (float) the learning rate
        iters - (int) the number of iterations
        """

    def predict_rating(self, user_id, movie_id):
        """
        INPUT:
        user_id - the user_id from the reviews df
        movie_id - the movie_id according the movies df

        OUTPUT:
        pred - the predicted rating for user_id-movie_id according to FunkSVD
        """

    def make_recommendations(self, _id, _id_type='movie', rec_num=5):
        """
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)
        rec_num - number of recommendations to return (int)

        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the
                       given movie, or recs for a user_id given
        """