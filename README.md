# Movie-Recommendation-Engine

Getting Started
---------------

This repository implements a Movie Recommendation Engine which recommends new movies 
to different users with FlunkSVD, Knowledge Based Recommendation and Content Based 
Recommendation.

Prerequisites
-------------

    numpy
    pandas

How to run the recommendation engine with the Recommender class?
----------------------------------------------------------------

Instantiate the recommender class
 
    rec = r.Recommender()
    
To make recommendations for users in the dataset, call the make_recommendations
function with first parameter as the user id and the second parameter as the keyword
'user'.

    rec.make_recommendations(8, 'user')  
    
To make recommendations for users not in the dataset, call the make_recommendations
function with first parameter as the user id and the second parameter as the keyword
'user'.   

    rec.make_recommendations(1, 'user')  

To make recommendations for a movie in the dataset, call the make_recommendations
function with the parameter as movie id.
 
    
    rec.make_recommendations(1853728)  
    
To make recommendations for a movie not in the dataset, call the make_recommendations
function with the parameter as movie id.    
    
    rec.make_recommendations(1)  


Summary
-------

Here, we have developed a Recommender System which uses FunkSVD to make predictions 
of user movie ratings. And uses either FunkSVD or a Knowledge Based Recommendation 
(highest ranked) to make recommendations for users.  Finally, if given a movie, 
the recommender will provide movies that are most similar as a Content Based Recommender.



