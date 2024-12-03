
import spacy
import requests
from bs4 import BeautifulSoup
import pytextrank
import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import re
from nltk.corpus import stopwords
import joblib

def veritabani():
    try:
        dt = psycopg2.connect(database="postgres",
                              user="postgres",
                              host='localhost',
                              password="12345",
                              port=5432)
        cr = dt.cursor()
        cr.execute("""
        CREATE TABLE IF NOT EXISTS yapay_zeka (
            metin_id SERIAL PRIMARY KEY,
            ana_konu VARCHAR(1000),
            alt_konu VARCHAR(1000),
            output TEXT
        );
        """)
        dt.commit()
        return dt, cr
    except Exception as e:
        print("Veritabani hatasi:", e)
        return None, None


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    stop_words = set(stopwords.words("turkish"))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)


class Siniflandirma:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer()
        self.model = LinearSVC()

    def egitim(self):
        data = pd.read_excel(self.data_path)
        data['metin'] = data['metin'].apply(preprocess_text)
        data_train, data_test = train_test_split(data, test_size=0.2, stratify=data['kategori'])
        x_train = self.vectorizer.fit_transform(data_train['metin'])
        x_test = self.vectorizer.transform(data_test['metin'])
        self.model.fit(x_train, data_train['kategori'])
        y_pred = self.model.predict(x_test)
        print("Model Performansi:")
        print(classification_report(data_test['kategori'], y_pred))
        # Model ve öznitelik çıkarıcıyı kaydet
        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.vectorizer, "vectorizer.pkl")

    def tahmin(self, metin):
        metin = preprocess_text(metin)
        vectorizer = joblib.load("vectorizer.pkl")
        model = joblib.load("model.pkl")
        x_new = vectorizer.transform([metin])
        return model.predict(x_new)[0]

def load_nlp_model():
    nlp = spacy.load("tr_core_news_md")
    nlp.add_pipe("textrank")
    return nlp

def ozetle(metin, nlp_model):
    doc = nlp_model(metin)
    return doc._.phrases[0].text if doc._.phrases else "Konu bulunamadi"

def aramamotoru(se):
    url = f"https://google.com/search?q={se}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    output = requests.get(url, headers=headers)
    if output.status_code == 200:
        soup = BeautifulSoup(output.text, "html.parser")
        sonuclar = []
        for result in soup.select(".tF2Cxc"):
            baslik = result.select_one("h3").text if result.select_one("h3") else "Başlik bulunamadi"
            snippet = result.select_one(".VwiC3b").text if result.select_one(".VwiC3b") else "Aciklama bulunamadi"
            sonuclar.append(f"Baslik: {baslik}\nSnippet: {snippet}\n")
        return sonuclar
    else:
        print(f"Hata oluştu: {output.status_code}")
        return []

def main():
    dt, cr = veritabani()
    if not dt or not cr:
        print("Veritabanina hatasi.")
        return

    data_path = "veri.xlsx"
    classifier = Siniflandirma(data_path)
    classifier.egitim()

    nlp_model = load_nlp_model()

    while True:
        ara = input("Aramak istediğiniz sorguyu girin (Cikis için 'q'): ").strip()
        if ara.lower() == "q":
            print("Program sonlandiriliyor.")
            break
        elif not ara:
            print("Boş bir sorgu girdiniz. Lütfen tekrar deneyin.")
            continue

        try:
            ana_konu = ozetle(ara, nlp_model)
            print(f"Ana Konu: {ana_konu}")
            alt_konu = classifier.tahmin(ana_konu)
            print(f"Alt Konu: {alt_konu}")

            sonuc = f"{ana_konu} {alt_konu}"
            output = aramamotoru(sonuc)
            output_text = " | ".join(output)[:1000]

            print(output_text)
            cr.execute(
                "INSERT INTO yapay_zeka (ana_konu, alt_konu, output) VALUES (%s, %s, %s)",
                (ana_konu, alt_konu, output_text)
            )
            dt.commit()
        except Exception as e:
            print("Hata:", e)
            dt.rollback()

    cr.close()
    dt.close()

if __name__ == "__main__":
    main()
