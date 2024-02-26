from flask import Flask, request, jsonify
from annoy import AnnoyIndex

app = Flask(__name__)
dim= 576
# Load the Annoy database
annoy_db = AnnoyIndex(dim, metric='angular')  # Here 2 is the dimension of the vectors in the database in my example
                                            # you would replace it with the dimension of your vectors
annoy_db.load('index.ann')  # Replace 'annoy_db.ann' with the path to your Annoy database


@app.route('/reco', methods=['POST']) # This route is used to get recommendations
def reco():
    vector = request.json['vector'] # Get the vector from the request
    reco = annoy_db.get_nns_by_vector(vector, 5) # Get the 2 closest elements indices
    return jsonify(reco) # Return the reco as a JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Run the server on port 5000 and make it accessible externally