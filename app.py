from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
mx = pickle.load(open('minmaxscaler.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    crop_dict = {1: "Riz", 2: "Maïs", 3: "Jute", 4: "Coton", 5: "Coco", 6: "Papaye", 7: "Orange",
                 8: "Pomme", 9: "Le melon (Cucumis melo)", 10: "Pasteque (Citrullus lanatus)", 11: "Raisins", 12: "Mangue", 13: "Banane",
                 14: "La grenade (Punica granatum)", 15: "Lentilles (Lens culinaris)", 16: "Haricot Noir (Phaseolus vulgaris)", 17: "soja vert (Vigna radiata)", 18: "Haricot Papillon(Vigna aconitifolia)",
                 19: "Pois Cajan (Cajanus cajan)", 20: "Haricots rouges", 21: "Pois chiche(Cicer arietinum)", 22: "Cafe"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} ".format(crop)
    else:
        result = "Désolé, nous n'avons pas pu déterminer la culture la plus adaptée avec les données fournies."
    return render_template('index.html',result = result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)