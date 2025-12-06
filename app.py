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
    "sampah", "ampas", "menipu", "tipu", "tipuan", "hoax", "palsu", "bohong", "pembohongan", "login", "gabisa","gimana"
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
# TAB TRAINING & EVALUASI (VERSI LENGKAP DENGAN PERHITUNGAN P(C), P(X|C), P(C|X))
# ============================================================
with tab2:
    if 'labeled_df' not in st.session_state:
        st.info("⚠ Silakan upload data dan lakukan labeling dulu di tab sebelumnya.")
    else:
        st.header("Training & Evaluasi Naive Bayes")

        # ============================
        # ✨ PENJELASAN TEORI
        # ============================
        st.subheader("📘 Dasar Teori Naive Bayes")
        st.markdown("""
Teorema Bayes:
\\[
P(C \\mid X) = \\frac{P(X \\mid C) \\cdot P(C)}{P(X)}
\\]

- **P(C)** → probabilitas awal kelas (prior)  
- **P(X|C)** → peluang kata muncul dalam kelas tertentu (likelihood)  
- **P(C|X)** → probabilitas kelas setelah melihat kata-kata (posterior)  

Naive Bayes pada teks:
\\[
P(C \\mid x_1 ... x_n) \\propto P(C) \\prod_i P(x_i \\mid C)
\\]
Menggunakan Laplace smoothing untuk mengatasi kata yang tidak muncul.
""")

        # =====================================
        # LOAD DATA
        # =====================================
        df = st.session_state['labeled_df'].copy()

        # =====================================
        # TRAIN MODEL
        # =====================================
        nb = NaiveBayesClassifier()
        nb.train(df)
        df['Prediksi'] = df['Preprocessed'].apply(nb.predict)

        # =====================================
        # 1️⃣ MENAMPILKAN PRIOR P(C)
        # =====================================
        st.subheader("📐 Probabilitas Awal Kelas (Prior)")

        total_data = len(df)
        jumlah_pos = (df['Label'] == 'Positif').sum()
        jumlah_neg = (df['Label'] == 'Negatif').sum()

        P_pos = jumlah_pos / total_data
        P_neg = jumlah_neg / total_data

        st.table(pd.DataFrame({
            'Kelas': ['Positif','Negatif'],
            'Jumlah Data': [jumlah_pos, jumlah_neg],
            'Prior P(C)': [P_pos, P_neg]
        }))

        # =====================================
        # 2️⃣ MENAMPILKAN LIKELIHOOD P(X|C)
        # =====================================
        st.subheader("🧩 Contoh Perhitungan Likelihood Kata (P(X|C))")

        all_words = " ".join(df['Preprocessed']).split()
        kata_teratas = [w for w, c in Counter(all_words).most_common(6)]

        rows = []
        for kata in kata_teratas:
            pos_tokens = " ".join(df[df['Label']=='Positif']['Preprocessed']).split()
            neg_tokens = " ".join(df[df['Label']=='Negatif']['Preprocessed']).split()

            freq_pos = pos_tokens.count(kata)
            freq_neg = neg_tokens.count(kata)

            total_pos = len(pos_tokens)
            total_neg = len(neg_tokens)

            likelihood_pos = (freq_pos + 1) / (total_pos + len(all_words))
            likelihood_neg = (freq_neg + 1) / (total_neg + len(all_words))

            rows.append([kata, freq_pos, freq_neg, likelihood_pos, likelihood_neg])

        st.table(pd.DataFrame(rows, columns=[
            "Kata",
            "Frekuensi Positif",
            "Frekuensi Negatif",
            "P(kata|Positif)",
            "P(kata|Negatif)"
        ]))

        # =====================================
        # 3️⃣ POSTERIOR P(C|X)
        # =====================================
        st.subheader("🎯 Contoh Perhitungan Posterior (P(C|X))")

        contoh_ulasan = st.text_input(
            "Masukkan contoh ulasan:",
            "aplikasinya bagus dan sangat membantu"
        )

        if contoh_ulasan:
            X = preprocess_text(contoh_ulasan, set(), set())
            tokens = X.split()

            pos_log = np.log(P_pos)
            neg_log = np.log(P_neg)

            pos_tokens = " ".join(df[df['Label']=='Positif']['Preprocessed']).split()
            neg_tokens = " ".join(df[df['Label']=='Negatif']['Preprocessed']).split()

            total_pos = len(pos_tokens)
            total_neg = len(neg_tokens)

            for kata in tokens:
                freq_pos = pos_tokens.count(kata)
                freq_neg = neg_tokens.count(kata)

                P_x_pos = (freq_pos + 1) / (total_pos + len(all_words))
                P_x_neg = (freq_neg + 1) / (total_neg + len(all_words))

                pos_log += np.log(P_x_pos)
                neg_log += np.log(P_x_neg)

            posterior_pos = np.exp(pos_log)
            posterior_neg = np.exp(neg_log)

            total = posterior_pos + posterior_neg

            st.table(pd.DataFrame({
                'Kelas': ['Positif','Negatif'],
                'P(C|X)': [posterior_pos/total, posterior_neg/total]
            }))

        # =====================================
        # 4️⃣ MENAMPILKAN HASIL PREDIKSI
        # =====================================
        st.subheader("📊 Hasil Prediksi Model")
        st.dataframe(df[['content','Preprocessed','Label','Prediksi']])

        # =====================================
        # 5️⃣ CONFUSION MATRIX
        # =====================================
        st.subheader("📉 Confusion Matrix")
        cm = confusion_matrix(df['Label'], df['Prediksi'], labels=['Positif','Negatif'])
        st.write(pd.DataFrame(cm,
            index=['Aktual Positif','Aktual Negatif'],
            columns=['Pred Positif','Pred Negatif']
        ))
        

        # =====================================
        # 6️⃣ METRIK EVALUASI (Satu Nilai Precision, Recall, F1)
        # =====================================
        st.subheader("🔍 Detail Evaluasi")
        
        TP = int(cm[0][0])
        FN = int(cm[0][1])
        FP = int(cm[1][0])
        TN = int(cm[1][1])
        
        accuracy = (TP + TN) / cm.sum()
        
        # Precision, Recall, F1 untuk kelas Positif (utama)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score_value = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        st.table(pd.DataFrame({
            'Metrik': [
                'True Positive (TP)', 
                'True Negative (TN)',
                'False Positive (FP)', 
                'False Negative (FN)',
                'Accuracy (%)',
                'Precision', 
                'Recall', 
                'F1-Score'
            ],
            'Nilai': [
                TP, TN, FP, FN,
                f"{accuracy*100:.2f}%",
                f"{precision:.4f}", 
                f"{recall:.4f}", 
                f"{f1_score_value:.4f}"
            ]
        }))








