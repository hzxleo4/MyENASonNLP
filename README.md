# An Implementation on sentimental analysis through ENAS
Code based on the paper https://arxiv.org/abs/1802.03268 and the code https://github.com/RualPerez/AutoML, You can find more details in https://github.com/RualPerez/AutoML.

## Difference between two repositories

Add three files:

- My_Policy_Gradient_AutoML.ipynb
- Mychildnet.py
- Generate_database.ipynb

Database can be download at https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews.

## How to run the code

- Download the database first
- Run the Generate_database.ipynb to get the input of the neutral network
- Run the My_Policy_Gradient_AutoML.ipynb to see the result.
- PS:a pretrained policy.pt can be used
