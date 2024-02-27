from flask import Flask, request, jsonify
from annoy import AnnoyIndex

app = Flask(__name__)
dim= 576
# Load the Annoy database
annoy_db = AnnoyIndex(dim, metric='angular')  
                                            
annoy_db.load('index.ann')  #'annoy_db.ann' is the path to the Annoy database


@app.route('/reco', methods=['POST']) # This route is used to get recommendations
def reco():
    vector = request.json['vector'] # Get the vector from the request
    reco = annoy_db.get_nns_by_vector(vector, 5) # Get the 5 closest elements indices
    return jsonify(reco) # Return the reco as a JSON

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Run the server on port 5000 and make it accessible externally
