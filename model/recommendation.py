import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, USER_COLLECTION

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
interactions_collection = db["interactions"]
sample_data = [
    {"userId": "user1", "productId": "productA", "rating": 5, "interaction_type": "purchase"},
    {"userId": "user2", "productId": "productB", "rating": 4, "interaction_type": "click"},
    {"userId": "user1", "productId": "productC", "rating": 3, "interaction_type": "view"},
    {"userId": "user3", "productId": "productA", "rating": 5, "interaction_type": "add_to_cart"},
    {"userId": "user2", "productId": "productC", "rating": 2, "interaction_type": "view"},
    {"userId": "user4", "productId": "productB", "rating": 4, "interaction_type": "purchase"},
    {"userId": "user5", "productId": "productD", "rating": 1, "interaction_type": "view"},
    {"userId": "user3", "productId": "productE", "rating": 5, "interaction_type": "purchase"},
    {"userId": "user5", "productId": "productC", "rating": 3, "interaction_type": "add_to_cart"},
    {"userId": "user4", "productId": "productA", "rating": 4, "interaction_type": "click"}
]

# Insert data into MongoDB
result = interactions_collection.insert_many(sample_data)
print(f"Inserted {len(result.inserted_ids)} documents.")


# Initialize MongoDB client

user_collection = db[USER_COLLECTION]

# Function to fetch user data from MongoDB
def get_user_data():
    """Fetch all user interaction data from MongoDB."""
    data = list(user_collection.find({}, {"_id": 0, "userId": 1, "productId": 1, "rating": 1, "interaction_type": 1}))
    if not data:
        return None

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Assign weights based on interaction type
    interaction_weights = {'view': 1, 'click': 2, 'add_to_cart': 3, 'purchase': 5}
    df['weighted_rating'] = df['rating'] * df['interaction_type'].map(interaction_weights)
    df['weighted_rating'] = pd.to_numeric(df['weighted_rating'], errors='coerce')


    return df

# Function to create the user-item matrix
def create_user_item_matrix(df):
    """Create a user-item matrix from the DataFrame."""
    user_item_matrix = df.pivot_table(
    index='userId',
    columns='productId',
    values='weighted_rating',
    aggfunc='mean').fillna(0)
    return csr_matrix(user_item_matrix.values), user_item_matrix

# Function to train the kNN model
def train_knn_model(matrix):
    """Train the kNN model on the user-item matrix."""
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3)
    model_knn.fit(matrix)
    return model_knn

# Function to get similar users using the trained kNN model
def get_similar_users(user_id, user_item_matrix, model_knn):
    """Find similar users to the given user ID."""
    if user_id not in user_item_matrix.index:
        return {"error": "User not found"}

    user_index = user_item_matrix.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(user_item_matrix.iloc[user_index].values.reshape(1, -1), n_neighbors=3)

    similar_users = []
    for i in range(1, len(distances.flatten())):
        similar_user_id = user_item_matrix.index[indices.flatten()[i]]
        similar_users.append(similar_user_id)

    return {"userId": user_id, "similarUsers": similar_users}
