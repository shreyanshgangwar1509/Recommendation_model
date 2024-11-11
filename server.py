from flask import Flask, jsonify
from model.recommendation import get_similar_users,get_user_data,create_user_item_matrix,train_knn_model
import pandas as pd
app = Flask(__name__)

@app.route('/recommendation/<user_id>', methods=['GET'])
def recommendation(user_id):
    # Fetch user interaction data from MongoDB
    df = get_user_data()
    if df is None:
        return jsonify({"error": "No user data found"}), 404

    # Create user-item matrix and train kNN model
    matrix, user_item_matrix = create_user_item_matrix(df)
    model_knn = train_knn_model(matrix)

    # Get similar users
    result = get_similar_users(user_id, user_item_matrix, model_knn)
    return jsonify(result)

# Route for members
@app.route('/members', methods=['GET'])
def members():
    return jsonify({"members": ["shreyansh", "Gangwar"]})

if __name__ == "__main__":
    app.run(debug=True)
