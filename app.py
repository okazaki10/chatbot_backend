from flask import Flask
from flask import request
from flask import jsonify
from keras.preprocessing.sequence import pad_sequences
import keras
import pickle

app = Flask(__name__)
model = keras.models.load_model('modelcnn.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def index():
    chat = request.args.get('chat', default="")
    maxlen = 100
    kolom = ['tag_0', 'tag_1', 'tag_2','tag_3']
    text_predict = [chat]
    seq = tokenizer.texts_to_sequences(text_predict)
    padded = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(padded)
    urutan = sorted(zip(pred[0], kolom), reverse=True)
    result = []
    for k, p in urutan:
        print(p, k)
        result.append({"tag": str(p), "accuracy": str(k)})
    data = {"data": result, "chat": chat}
    return jsonify(data)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    app.run(debug=True)