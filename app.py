from flask import Flask
from flask import request
from flask import jsonify
from keras.preprocessing.sequence import pad_sequences
import keras
import pickle
import requests
import pandas as pd
import io
import re

url="https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"

# mendownload dataset
s=requests.get(url).content

# membuka dataset
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
c

def alay(text):
  for s,f in zip(c['slang'],c['formal']):
    if text == s:
      text = f
      return text
  return text

def ubah_alay(text):    
    text = [alay(word) for word in text.split()]
    return " ".join(text)

# daftar kata yang tidak penting
stopwords = ["bu","halo","dok","selamat","terima kasih","pagi","siang","sore","malam",
             "assalamualaikum","wr","wb","assalamu","alaikum","pak","dokter",
             "mau","tanya","nanya","bertanya","menanyakan","perempuan","laki-laki","laki laki"]

def hapus(text):
  if text in stopwords:
    return ""
  return text

def ubah_kata(text):    
    text = [hapus(word) for word in text.split()]
    return " ".join(text)

app = Flask(__name__)
model = keras.models.load_model('modelcnn.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def index():
    chat = request.args.get('chat', default="")
    maxlen = 100
    kolom = ['tag_bipolar','tag_depresi','tag_insomnia','tag_kecemasan','tag_skizofrenia']
    # praproses
    # Melakukan lower casing
    chat = chat.lower()

    # Menghapus tanda baca
    chat = re.sub(r'[^a-zA-Z0-9]',' ',chat)

    # Menghilangkan karakter tunggal
    chat = re.sub('\s+[a-zA-Z]\s+',' ',chat)

    # Menghilagkan spasi ganda
    chat = re.sub('[•\t|\n|\s+]',' ',chat)

    # mengubah kata pada text chat ke bahasa yang lebih formal
    chat = ubah_alay(chat)

    # menghapus kata yang tidak penting
    chat = ubah_kata(chat)

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