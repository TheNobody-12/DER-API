
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB4
app = Flask(__name__)
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
model = load_model('Efficientmodel1.h5', compile=False)
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
]
lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 20,verbose = 1,factor = 0.50, min_lr = 1e-10)

mcp = ModelCheckpoint('model.h5')

es = EarlyStopping(verbose=1, patience=20)

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=METRICS)
def Predict_emotion(filepath):
    i = tf.keras.utils.load_img(filepath, target_size=(224,224))
    i= tf.keras.utils.img_to_array(i)/255.0
    i = i.reshape(1, 224,224,3)
    y_pred = model.predict(i)
    predicted_classes = tf.argmax(y_pred, axis=1)
    category_names = ["angry", "happy", "relaxed", "sad"] 
    predicted_categories = [category_names[value] for value in predicted_classes]
    return predicted_categories[0]

@app.route("/")
def main():
    return render_template("home.html")

@app.route("/reset",methods=["POST"])
def reset():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img = request.files["imagefile"]
        img_path = "static/" + img.filename
        img.save(img_path)
        p = Predict_emotion(img_path)
        return render_template("home.html", prediction=p, img_path=img_path)

if __name__ == "__main__":
    app.run(debug=True)








    


    



