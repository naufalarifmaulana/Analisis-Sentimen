import streamlit as st
import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import io

# Coba import google_play_scraper; bila tidak ada, jangan crash — berikan pesan
HAVE_SCRAPER = True
try:
    from google_play_scraper import reviews, Sort
except Exception as e:
    HAVE_SCRAPER = False
    _IMPORT_ERROR = str(e)

# ---------------------------
# Fungsi Preprocessing
# ---------------------------
def load_stopwords(file):
    words = file.read().decode('utf-8').splitlines()
    return set(words)

def preprocess_text(text, kamus, sentimen_words):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in kamus or t in sentimen_words]
    return ' '.join(tokens)

# ---------------------------
# Kata Positif & Negatif + Labeling
# (gunakan listmu — saya ringkas agar tidak terlalu panjang di sini)
# ---------------------------
positive_words = [
    "baik","bagus","cepat","mudah","enak","nyaman","puas","suka","senang","aman",
    "stabil","responsif","efektif","efisien","berguna","bermanfaat" "baik",
"bagus","cepat","mudah","enak","nyaman","puas","suka","senang","aman","stabil","responsif","efektif","efisien","berguna","bermanfaat","maksimal","optimal","akurat","cocok","ramah","jelas","sesuai","bebas","menyenangkan","menarik","keren","hebat","mantap","unggul","positif","oke","solid","top","sempurna"

]
negative_words = [
    "buruk", "jelek", "error", "eror", "gagal", "lemot", "lelet", "lambat", "lola", "slow",
    "parah", "susah", "lama", "kecewa", "kacau", "update", "hang", "bug", "crash", "bermasalah",
    "kurang", "tidak bisa", "tidak mau", "tidak berfungsi", "tidak jalan", "tidak bekerja",
    "macet", "freeze", "terhenti", "hilang", "terhapus", "ngelag", "lag", "ngehang", "ribet",
    "berantakan", "membingungkan", "menyusahkan", "payah", "jelek sekali", "tidak memuaskan",
    "tidak responsif", "membosankan", "menjengkelkan", "menyebalkan", "mengecewakan", "menakutkan",
    "tidak jelas", "tidak sesuai", "tidak stabil", "mati", "shutdown", "restart", "resiko",
    "hancur", "rusak", "cacat", "tidak enak", "tidak nyaman", "melelahkan", "membuat marah",
    "tidak ramah", "tidak berguna", "tidak efektif", "tidak efisien", "menurunkan", "memburuk",
    "lemah", "boros", "tidak aman", "keterlaluan", "terlalu lambat", "terlalu lama", "terlalu rumit",
    "bikin stress", "stress", "males", "muak", "benci", "jijik", "frustrasi", "sangat buruk",
    "tidak worth it", "tidak bermanfaat", "tidak ada gunanya", "tidak bekerja dengan baik",
    "terbuang", "buang waktu", "tidak maksimal", "tidak optimal", "tidak akurat", "tidak cocok",
    "sampah", "ampas", "menipu", "tipu", "tipuan", "hoax", "palsu", "bohong", "pembohongan"
]

def auto_label(text, positive_words, negative_words):
    text = str(text).lower()
    for phrase in negative_words:
        if re.search(r'\b' + re.escape(phrase) + r'\b', text):
            return 'Negatif'
    for phrase in positive_words:
        if re.search(r'\b' + re.escape(phrase) + r'\b', text):
            return 'Positif'
    return 'Positif'

# ---------------------------
# Naive Bayes Classifier (sama seperti milikmu)
# ---------------------------
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}
        self.vocab = set()
    
    def train(self, data, label_col='Label', text_col='Preprocessed'):
        self.vocab = set()
        counts = {'Positif': Counter(), 'Negatif': Counter()}
        
        for _, row in data.iterrows():
            label = row[label_col]
            tokens = str(row[text_col]).split()
            counts[label].update(tokens)
            self.vocab.update(tokens)
        
        total_docs = len(data)
        # keamanan: bila salah satu kelas kosong, tambahkan smoothing kecil
        for c in ['Positif','Negatif']:
            if len(data[data[label_col]==c]) == 0:
                self.class_probs[c] = np.log(1e-9)
            else:
                self.class_probs[c] = np.log(len(data[data[label_col]==c])/total_docs)
        
        self.word_probs = {c:{} for c in ['Positif','Negatif']}
        for c in ['Positif','Negatif']:
            total_words = sum(counts[c].values())
            for word in self.vocab:
                self.word_probs[c][word] = np.log((counts[c][word] + 1)/(total_words + len(self.vocab)))
    
    def predict(self, text):
        tokens = str(text).split()
        scores = {c:self.class_probs.get(c, np.log(1e-9)) for c in ['Positif','Negatif']}
        for c in ['Positif','Negatif']:
            for t in tokens:
                if t in self.word_probs[c]:
                    scores[c] += self.word_probs[c][t]
        return max(scores, key=scores.get)

# ------------------------------------------------------------
# UI STREAMLIT
# ------------------------------------------------------------
st.title("📊 Sistem Analisis Sentimen Naive Bayes")

tab0, tab1, tab2 = st.tabs([
    "📥 Scraping Google Play",
    "📝 Preprocessing Data",
    "🤖 Training & Evaluasi"
])

# ============================================================
# TAB SCRAPING GOOGLE PLAY
# ============================================================
with tab0:
    st.header("Scraping Ulasan Google Play")
    if not HAVE_SCRAPER:
        st.warning(
            "Module `google_play_scraper` tidak ditemukan di environment ini.\n"
            "Untuk mengaktifkan fitur scraping, install package terlebih dahulu:\n\n"
            "pip install google-play-scraper\n\n"
            "Jika Anda menggunakan Streamlit Cloud atau hosting lain, tambahkan `google-play-scraper` ke requirements.txt dan redeploy."
        )
        st.write("Detail error import:", _IMPORT_ERROR)
        st.info("Setelah menginstall, jalankan ulang aplikasinya.")
    else:
        app_id = st.text_input("Masukkan App ID (contoh: com.siabang.simantap)")
        jumlah = st.number_input("Jumlah ulasan yang ingin di-scrape", min_value=10, max_value=500, value=100, step=10)
        jumlah = int(jumlah)

        if st.button("Mulai Scraping"):
            if not app_id:
                st.error("App ID kosong — isi dulu (contoh: com.example.app).")
            else:
                try:
                    with st.spinner("Mengambil data dari Google Play..."):
                        data, _ = reviews(
                            app_id,
                            lang='id',
                            country='id',
                            sort=Sort.NEWEST,
                            count=jumlah
                        )
                    df_scrape = pd.DataFrame(data)[['userName','content','score']]
                    st.success("Scraping berhasil! Di sini hanya tersedia preview dan tombol download (tidak otomatis masuk ke preprocessing).")
                    st.dataframe(df_scrape)

                    csv = df_scrape.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "💾 Download Hasil Scraping (CSV)",
                        data=csv,
                        file_name="hasil_scraping_google_play.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat scraping: {e}")

# ============================================================
# TAB PREPROCESSING
# ============================================================
with tab1:
    st.header("Upload & Preprocessing")
    uploaded_csv = st.file_uploader("📂 Upload CSV ulasan (mis. hasil scraping)", type='csv')
    uploaded_kamus = st.file_uploader("📂 Upload kamus stopwords (TXT)", type='txt')

    def read_csv_auto(file):
        try:
            df = pd.read_csv(file, engine='python')
            if 'content' not in df.columns:
                st.error("CSV harus memiliki kolom 'content'.")
                return None
            return df
        except Exception as e:
            st.error(f"Error membaca CSV: {e}")
            return None

    if uploaded_csv and uploaded_kamus:
        df = read_csv_auto(uploaded_csv)
        if df is not None:
            kamus = load_stopwords(uploaded_kamus)
            sentimen_words = set(positive_words + negative_words)

            df['Preprocessed'] = df['content'].apply(lambda x: preprocess_text(x, kamus, sentimen_words))
            df['Label'] = df['Preprocessed'].apply(lambda x: auto_label(x, positive_words, negative_words))

            st.subheader("Edit Label Jika Ingin:")
            edited_df = st.data_editor(
                df[['content','Preprocessed','Label']],
                column_config={
                    "Label": st.column_config.SelectboxColumn(
                        "Label",
                        options=["Positif","Negatif"]
                    )
                },
                num_rows="dynamic"
            )

            if st.button("Simpan Hasil Labeling"):
                # simpan ke session agar bisa training
                st.session_state['labeled_df'] = edited_df.copy()
                # juga simpan file lokal (opsional)
                edited_df.to_csv("labeled_data.csv", index=False)
                st.success("Labeling disimpan sebagai labeled_data.csv!")

# ============================================================
# TAB TRAINING & EVALUASI
# ============================================================
with tab2:
    if 'labeled_df' not in st.session_state:
        st.info("⚠ Silakan upload data (hasil scraping CSV) dan lakukan labeling dulu di tab 'Upload & Preprocessing'.")
    else:
        st.header("Training & Evaluasi Naive Bayes")
        df = st.session_state['labeled_df'].copy()

        # pastikan kolom Preprocessed ada (jika user mengedit layout)
        if 'Preprocessed' not in df.columns:
            st.error("Data tidak memiliki kolom 'Preprocessed'. Kembali ke tab Preprocessing.")
        else:
            nb = NaiveBayesClassifier()
            nb.train(df)

            df['Prediksi'] = df['Preprocessed'].apply(nb.predict)

            st.subheader("Hasil Prediksi")
            st.dataframe(df[['content','Label','Prediksi']])

            cm = confusion_matrix(df['Label'], df['Prediksi'], labels=['Positif','Negatif'])
            st.subheader("Confusion Matrix")
            st.write(pd.DataFrame(cm, index=['Positif','Negatif'], columns=['Positif','Negatif']))

            TP = int(cm[0][0])
            FN = int(cm[0][1])
            FP = int(cm[1][0])
            TN = int(cm[1][1])

            accuracy = (TP + TN) / cm.sum() if cm.sum() > 0 else 0.0
            precision_pos = precision_score(df['Label'], df['Prediksi'], pos_label='Positif', zero_division=0)
            recall_pos = recall_score(df['Label'], df['Prediksi'], pos_label='Positif', zero_division=0)
            f1_pos = f1_score(df['Label'], df['Prediksi'], pos_label='Positif', zero_division=0)
            precision_neg = precision_score(df['Label'], df['Prediksi'], pos_label='Negatif', zero_division=0)
            recall_neg = recall_score(df['Label'], df['Prediksi'], pos_label='Negatif', zero_division=0)
            f1_neg = f1_score(df['Label'], df['Prediksi'], pos_label='Negatif', zero_division=0)

            st.subheader("Detail Evaluasi")
            st.table(pd.DataFrame({
                'Metrik': ['TP','TN','FP','FN','Accuracy','Precision+','Recall+','F1+','Precision-','Recall-','F1-'],
                'Nilai': [TP,TN,FP,FN,f"{accuracy*100:.2f}%",
                          f"{precision_pos:.2f}", f"{recall_pos:.2f}", f"{f1_pos:.2f}",
                          f"{precision_neg:.2f}", f"{recall_neg:.2f}", f"{f1_neg:.2f}"]
            }))

            # Export RapidMiner
            export_df = df[['content','Preprocessed','Label','Prediksi']]
            st.download_button(
                "📥 Download untuk RapidMiner (CSV)",
                export_df.to_csv(index=False).encode('utf-8'),
                file_name="data_rapidminer.csv",
                mime="text/csv"
            )

