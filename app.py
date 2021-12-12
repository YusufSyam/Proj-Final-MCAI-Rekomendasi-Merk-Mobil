from flask import Flask, render_template, request
from prediction import pred

import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods = ['GET', "POST"])
def result():
    if request.method == 'POST':
        cc = int(request.form["cc"])
        transmisi = request.form["transmisi"]
        bahan_bakar = request.form["bahan_bakar"]
        kapasitas_tangki = int(request.form["kapasitas_tangki"])
        berat = int(request.form["berat"])
        jok = int(request.form["jok"])
        sistem_setir = request.form["sistem_setir"]
        jumlah_pintu = int(request.form["jumlah_pintu"])
        jumlah_roda = int(request.form["jumlah_roda"])
        panjang = int(request.form["panjang"])
        lebar = int(request.form["lebar"])
        tinggi = int(request.form["tinggi"])

        recommend= pred([cc,transmisi,bahan_bakar,kapasitas_tangki,berat,jok,sistem_setir,jumlah_pintu,jumlah_roda,panjang,lebar,tinggi])[0]
        recommend_car= 'Rekomendasi Tipe Mobil: '+recommend.split('|||')[0].capitalize()
        recommend_car_brand= 'Rekomendasi Brand / Model: '+ recommend.split('|||')[1].capitalize()

        return render_template("index.html", result_car=  recommend_car, result_model_brand= recommend_car_brand, p_cc=cc,p_transmisi=transmisi,p_bahan_bakar=bahan_bakar,p_kapasitas_tangki=kapasitas_tangki,p_berat=berat,p_jok=jok,p_sistem_setir=sistem_setir,p_jumlah_pintu=jumlah_pintu,p_jumlah_roda=jumlah_roda,p_panjang=panjang,p_lebar=lebar,p_tinggi=tinggi)

    else:
        return render_template("index.html", result_car=  '', result_model_brand= '', p_cc='',p_transmisi='',p_bahan_bakar='',p_kapasitas_tangki='',p_berat='',p_jok='',p_sistem_setir='',p_jumlah_pintu='',p_jumlah_roda='',p_panjang='',p_lebar='',p_tinggi='')

if __name__ == '__main__':

    app.run(debug=True)