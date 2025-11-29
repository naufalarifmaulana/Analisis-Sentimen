import streamlit as st
import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import io

# ---------------------------
# Fungsi Preprocessing
# ---------------------------
def load_stopwords(file):
    words = file.read().decode('utf-8').splitlines()
    return set(words)

def preprocess_text(text, kamus, sentimen_words):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)  # hapus simbol
    tokens = text.split()
    # Hapus stopwords, tapi jaga kata sentimen
    tokens = [t for t in tokens if t not in kamus or t in sentimen_words]
    return ' '.join(tokens)

# ---------------------------
# Fungsi Labeling Otomatis
# ---------------------------
positive_words = [
    "baik", "bagus", "cepat", "mudah", "enak", "nyaman", "puas", "suka", "senang", "aman",
    "stabil", "responsif", "efektif", "efisien", "berguna", "bermanfaat", "maksimal", "optimal",
    "akurat", "cocok", "ramah", "jelas", "sesuai", "bebas", "aman", "menyenangkan", "menarik",
    "keren", "hebat", "luar biasa", "sangat baik", "worth it", "bekerja dengan baik", "tidak ada masalah",
    "tidak buruk", "tidak jelek", "tidak lambat", "tidak susah", "tidak lama", "tidak kecewa",
    "tidak kacau", "tidak hang", "tidak bug", "tidak crash", "tidak bermasalah", "tidak kurang",
    "tidak macet", "tidak freeze", "tidak terhenti", "tidak hilang", "tidak terhapus", "tidak ngelag",
    "tidak lag", "tidak ngehang", "tidak ribet", "tidak berantakan", "tidak membingungkan",
    "tidak menyusahkan", "tidak payah", "tidak jelek sekali", "tidak tidak memuaskan",
    "tidak tidak responsif", "tidak membosankan", "tidak menjengkelkan", "tidak menyebalkan",
    "tidak mengecewakan", "tidak menakutkan", "tidak tidak jelas", "tidak tidak sesuai",
    "tidak tidak stabil", "tidak mati", "tidak shutdown", "tidak restart", "tidak resiko",
    "tidak hancur", "tidak rusak", "tidak cacat", "tidak tidak enak", "tidak tidak nyaman",
    "tidak melelahkan", "tidak membuat marah", "tidak tidak ramah", "tidak tidak berguna",
    "tidak tidak efektif", "tidak tidak efisien", "tidak menurunkan", "tidak memburuk",
    "tidak lemah", "tidak boros", "tidak tidak aman", "tidak keterlaluan", "tidak terlalu lambat",
    "tidak terlalu lama", "tidak terlalu rumit", "tidak bikin stress", "tidak stress", "tidak males",
    "tidak muak", "tidak benci", "tidak jijik", "tidak frustrasi", "tidak sangat buruk",
    "tidak tidak worth it", "tidak tidak bermanfaat", "tidak tidak ada gunanya",
    "tidak tidak bekerja dengan baik", "tidak terbuang", "tidak buang waktu", "tidak tidak maksimal",
    "tidak tidak optimal", "tidak tidak akurat", "tidak tidak cocok", "tidak sampah", "tidak ampas",
    "tidak menipu", "tidak tipu", "tidak tipuan", "tidak hoax", "tidak palsu", "tidak bohong", "tidak pembohongan"
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
    # Prioritas negatif: jika ada kata negatif, langsung negatif
    for phrase in negative_words:
        if re.search(r'\b' + re.escape(phrase) + r'\b', text):
            return 'Negatif'
    # Jika tidak ada negatif, cek positif
    for phrase in positive_words:
        if re.search(r'\b' + re.escape(phrase) + r'\b', text):
            return 'Positif'
    # Default positif jika tidak ada kata sentimen
    return 'Positif'

# ---------------------------
# Fungsi Naive Bayes
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
            tokens = row[text_col].split()
            counts[label].update(tokens)
            self.vocab.update(tokens)
        
        total_docs = len(data)
        self.class_probs = {c: np.log(len(data[data[label_col]==c])/total_docs) for c in ['Positif','Negatif']}
        
        self.word_probs = {c:{} for c in ['Positif','Negatif']}
        for c in ['Positif','Negatif']:
            total_words = sum(counts[c].values())
            for word in self.vocab:
                self.word_probs[c][word] = np.log((counts[c][word] + 1)/(total_words + len(self.vocab)))
    
    def predict(self, text):
        tokens = text.split()
        scores = {c:self.class_probs[c] for c in ['Positif','Negatif']}
        for c in ['Positif','Negatif']:
            for t in tokens:
                if t in self.word_probs[c]:
                    scores[c] += self.word_probs[c][t]
        return max(scores, key=scores.get)

# ---------------------------
# Streamlit App
# ---------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.5em;
        font-weight: bold;
        color: #2196F3;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Sistem Analisis Sentimen dengan Naive Bayes</div>', unsafe_allow_html=True)

# Tabs untuk navigasi
tab1, tab2 = st.tabs(["📝 Preprocessing Data", "🤖 Training & Evaluasi"])

with tab1:
    st.markdown('<div class="section-header">Upload dan Preprocessing</div>', unsafe_allow_html=True)
    st.markdown("**Pilih file CSV data ulasan Anda (pastikan ada kolom 'content'):**")
    uploaded_csv = st.file_uploader("📂 Unggah CSV Ulasan", type='csv', label_visibility="visible")
    
    st.markdown("**Pilih file TXT kamus stopwords Anda:**")
    uploaded_kamus = st.file_uploader("📂 Unggah Kamus Stopwords", type='txt', label_visibility="visible")


    def read_csv_auto(uploaded_csv):
        try:
            df = pd.read_csv(uploaded_csv, sep=None, engine='python')
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            return None
        if 'content' not in df.columns:
            st.error(f"CSV harus memiliki kolom 'content'. Kolom tersedia: {list(df.columns)}")
            return None
        return df

    if uploaded_csv and uploaded_kamus:
        df = read_csv_auto(uploaded_csv)
        if df is not None:
            kamus = load_stopwords(uploaded_kamus)
            sentimen_words = set(positive_words + negative_words)  # Gabungkan dan buat set
            df['Preprocessed'] = df['content'].apply(lambda x: preprocess_text(str(x), kamus, sentimen_words))
            df['Label'] = df['Preprocessed'].apply(lambda x: auto_label(x, positive_words, negative_words))

            st.subheader("Preview Data Labeling (bisa diedit semua)")
            # Konfigurasi kolom untuk dropdown pada Label
            column_config = {
                "Label": st.column_config.SelectboxColumn(
                    "Label",
                    options=["Positif", "Negatif"],
                    required=True,
                )
            }
            edited_df = st.data_editor(df[['content','Preprocessed','Label']], column_config=column_config, num_rows="dynamic")
            
            if st.button("Simpan Hasil Labeling"):
                edited_df.to_csv("labeled_data.csv", index=False)
                st.success("Data berhasil disimpan sebagai labeled_data.csv")
                st.session_state['labeled_df'] = edited_df

with tab2:
    if 'labeled_df' in st.session_state:
        st.markdown('<div class="section-header">Training & Evaluasi Naive Bayes</div>', unsafe_allow_html=True)
        df = st.session_state['labeled_df']

        nb = NaiveBayesClassifier()
        nb.train(df)

        df['Prediksi'] = df['Preprocessed'].apply(nb.predict)

        st.subheader("Hasil Prediksi")
        st.dataframe(df[['content','Label','Prediksi']], height=400)

        # Confusion Matrix
        cm = confusion_matrix(df['Label'], df['Prediksi'], labels=['Positif','Negatif'])
        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(cm, index=['Positif','Negatif'], columns=['Positif','Negatif'])
        st.write(cm_df)

        # Ekstrak TP, TN, FP, FN
        TP = cm[0][0]
        FN = cm[0][1]
        FP = cm[1][0]
        TN = cm[1][1]

        # Metrik
        accuracy = (TP + TN) / cm.sum()
        precision_pos = precision_score(df['Label'], df['Prediksi'], pos_label='Positif')
        recall_pos = recall_score(df['Label'], df['Prediksi'], pos_label='Positif')
        f1_pos = f1_score(df['Label'], df['Prediksi'], pos_label='Positif')

        precision_neg = precision_score(df['Label'], df['Prediksi'], pos_label='Negatif')
        recall_neg = recall_score(df['Label'], df['Prediksi'], pos_label='Negatif')
        f1_neg = f1_score(df['Label'], df['Prediksi'], pos_label='Negatif')

        # Tampilkan dalam tabel untuk rapat
        st.subheader("Detail Evaluasi")
        eval_df = pd.DataFrame({
            'Metrik': ['TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision (Pos)', 'Recall (Pos)', 'F1 (Pos)', 'Precision (Neg)', 'Recall (Neg)', 'F1 (Neg)'],
            'Nilai': [TP, TN, FP, FN, f"{accuracy*100:.2f}%", f"{precision_pos:.2f}", f"{recall_pos:.2f}", f"{f1_pos:.2f}", f"{precision_neg:.2f}", f"{recall_neg:.2f}", f"{f1_neg:.2f}"]
        })
        st.table(eval_df)

        # Ekspor untuk RapidMiner
        export_df = df[['content', 'Preprocessed', 'Label', 'Prediksi']]
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data untuk RapidMiner (CSV)",
            data=csv_data,
            file_name="data_untuk_rapidminer.csv",
            mime="text/csv"
        )
    else:
        st.info("Upload data dan simpan labeling terlebih dahulu untuk melihat evaluasi.")
