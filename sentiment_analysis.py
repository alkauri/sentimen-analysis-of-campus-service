"""
Program: Analisis Sentimen Layanan Kampus (Indonesia)

Deskripsi singkat:
- Membaca dataset CSV berisi teks (tweet/ulasan) tentang layanan kampus
- Melakukan preprocessing: pembersihan, tokenisasi, penghapusan stopword, stemming (Sastrawi)
- Opsi pemodelan:
  1) Pelatihan model klasik (Logistic Regression / Naive Bayes) bila label tersedia
  2) Inferensi dengan model pra-latih (IndoBERT/Roberta) bila tidak ada label
- Evaluasi (jika train/test): akurasi, precision, recall, F1, confusion matrix
- Menyimpan hasil klasifikasi ke CSV dan menampilkan grafik distribusi sentimen

Cara pakai ringkas:
python sentiment_analysis.py --csv unib.csv --text-col content --label-col label --model lr --test-size 0.2 --output hasil_sentimen.csv
"""

import argparse
import os
# Set backend Matplotlib non-interaktif agar kode dapat berjalan di lingkungan tanpa GUI
# (grafik akan disimpan/ditampilkan tanpa membuka jendela terpisah)
os.environ["MPLBACKEND"] = "agg"
import re
import sys
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sastrawi for Indonesian stemming and stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


def build_indonesian_stopwords(extra_stopwords: Optional[List[str]] = None) -> set:
    """Membangun himpunan stopword Bahasa Indonesia.

    - Menggunakan daftar stopword dari Sastrawi
    - Dapat ditambah kata-kata domain kampus (mis. "siakad", "layanan", dll.)
    - `extra_stopwords` memungkinkan penambahan dari argumen eksternal
    """
    factory = StopWordRemoverFactory()
    stopwords = set(factory.get_stop_words())
    domain_words = {
        "kampus",
        "univ",
        "universitas",
        "kampusnya",
        "fakultas",
        "akademik",
        "administrasi",
        "fasilitas",
        "sistem",
        "informasi",
        "simak",
        "siakad",
        "layanan",
        "mahasiswa",
        "mhs",
        "biro",
        "bagian",
        "unit",
        "pelayanan",
    }
    stopwords.update(domain_words)
    if extra_stopwords:
        stopwords.update(map(str.lower, extra_stopwords))
    return stopwords


def compile_regex_patterns():
    """Menyusun pola regex untuk pembersihan teks.

    Pola yang dihapus: URL, mention, hashtag, angka, emoji, tanda baca.
    Mengembalikan kumpulan pola untuk digunakan pada fungsi `clean_text`.
    """
    url_pattern = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    mention_pattern = re.compile(r"@\w+")
    hashtag_pattern = re.compile(r"#\w+")
    number_pattern = re.compile(r"\d+")
    # Basic emoji pattern (covers most common ranges)
    emoji_pattern = re.compile(
        "[\U0001F1E0-\U0001F1FF]"  # flags (iOS)
        "|[\U0001F300-\U0001F5FF]"  # symbols & pictographs
        "|[\U0001F600-\U0001F64F]"  # emoticons
        "|[\U0001F680-\U0001F6FF]"  # transport & map symbols
        "|[\U0001F700-\U0001F77F]"  # alchemical symbols
        "|[\U0001F780-\U0001F7FF]"
        "|[\U0001F800-\U0001F8FF]"
        "|[\U0001F900-\U0001F9FF]"
        "|[\U0001FA00-\U0001FA6F]"
        "|[\U0001FA70-\U0001FAFF]"
        "|[\U00002700-\U000027BF]"  # Dingbats
        "|[\U00002600-\U000026FF]",
        flags=re.UNICODE,
    )
    punctuation_pattern = re.compile(r"[^a-z\s]")  # after lowering, keep only a-z and spaces
    return url_pattern, mention_pattern, hashtag_pattern, number_pattern, emoji_pattern, punctuation_pattern


def clean_text(text: str, regexes) -> str:
    """Membersihkan teks mentah menjadi huruf kecil dan hanya alfabet a-z.

    Langkah:
    - lowercasing
    - hapus URL/mention/hashtag/emoji/angka
    - hapus karakter non-huruf (tanda baca), rapikan spasi
    """
    url_pattern, mention_pattern, hashtag_pattern, number_pattern, emoji_pattern, punctuation_pattern = regexes
    text = str(text)
    text = text.lower()
    text = url_pattern.sub(" ", text)
    text = mention_pattern.sub(" ", text)
    text = hashtag_pattern.sub(" ", text)
    text = emoji_pattern.sub(" ", text)
    text = number_pattern.sub(" ", text)
    # Replace non-letters with space, collapse whitespace
    text = punctuation_pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_stop_stem(text: str, stemmer, stopwords_set: set) -> str:
    """Tokenisasi sederhana + stopword removal + stemming Sastrawi.

    Mengembalikan teks yang telah dinormalisasi untuk vektorisasi TF-IDF.
    """
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_set and len(t) > 1]
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)


def preprocess_series(series: pd.Series) -> pd.Series:
    """Preprocessing masal pada kolom teks: cleaning → tokenisasi → stopword → stemming."""
    regexes = compile_regex_patterns()
    stemmer = StemmerFactory().create_stemmer()
    stopwords_set = build_indonesian_stopwords()

    # Apply cleaning, tokenization, stopword removal, and stemming
    cleaned = series.fillna("").map(lambda x: clean_text(x, regexes))
    processed = cleaned.map(lambda x: tokenize_stop_stem(x, stemmer, stopwords_set))
    return processed


def ensure_label_column(y: pd.Series) -> Tuple[pd.Series, dict]:
    """Menormalkan label sentimen ke {"negative","neutral","positive"}.

    - Menerima variasi umum Indonesia/Inggris (positif/positive/pos, negatif/negative/neg, netral/neutral)
    - Label di luar 3 kelas akan dianggap tidak valid (NaN)
    - Mengembalikan seri label ternormalisasi dan peta mapping yang dipakai
    """
    mapping = {
        "positif": "positive",
        "positive": "positive",
        "pos": "positive",
        "+": "positive",
        "negatif": "negative",
        "negative": "negative",
        "neg": "negative",
        "-": "negative",
        "netral": "neutral",
        "neutral": "neutral",
        "neu": "neutral",
        "0": "neutral",
    }

    def normalize(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        return mapping.get(s, s)

    y_norm = y.map(normalize)
    # Keep only three known classes
    valid = {"negative", "neutral", "positive"}
    y_norm = y_norm.where(y_norm.isin(valid), np.nan)
    return y_norm, mapping


def train_and_evaluate_ml(
    texts: pd.Series,
    labels: pd.Series,
    model_name: str = "lr",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Melatih model klasik (LR/NB) dan mengevaluasi pada test set.

    - Split train/test terstratifikasi
    - TF-IDF ngram (1-2) sebagai representasi fitur
    - Model default: Logistic Regression; alternatif: Multinomial Naive Bayes
    - Mengembalikan: model, vectorizer, y_test, y_pred, classification_report (dict), confusion_matrix
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if model_name == "nb":
        model = MultinomialNB()
    else:
        model = LogisticRegression(max_iter=200, solver="lbfgs")

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    report = classification_report(y_test, y_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
    return model, vectorizer, y_test, y_pred, report, cm


def try_load_indobert_pipeline():
    """Mencoba memuat pipeline klasifikasi sentimen pra-latih (IndoBERT/Roberta).

    Catatan: membutuhkan paket `transformers` dan `torch`, serta koneksi internet
    saat pertama kali mengunduh model.
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

        model_id = "w11wo/indonesian-roberta-base-sentiment-classifier"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
        return pipe
    except Exception as e:
        print("Gagal memuat model IndoBERT/Roberta untuk sentimen:", e)
        return None


def predict_with_bert(pipe, texts: List[str]) -> List[str]:
    """Melakukan prediksi sentimen menggunakan pipeline model pra-latih.

    Mengembalikan label normalized: negative/neutral/positive.
    """
    preds = pipe(texts, truncation=True)
    # Model labels typically are: LABEL_0, LABEL_1, LABEL_2 or explicit strings
    out = []
    for p in preds:
        label = p.get("label", "")
        lab = str(label).lower()
        if "neg" in lab:
            out.append("negative")
        elif "pos" in lab:
            out.append("positive")
        elif "neu" in lab or "netral" in lab or "neutral" in lab:
            out.append("neutral")
        else:
            # fallback using score sign if available; otherwise neutral
            out.append("neutral")
    return out


def plot_distribution(labels: pd.Series, title: str, output_path: Optional[str] = None):
    """Menampilkan dan (opsional) menyimpan grafik distribusi kelas sentimen."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels, order=["negative", "neutral", "positive"], palette=["#d9534f", "#f0ad4e", "#5cb85c"]) 
    plt.title(title)
    plt.xlabel("Sentimen")
    plt.ylabel("Jumlah")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def print_classification_report(report_dict: dict):
    """Mencetak metrik utama per kelas serta akurasi keseluruhan."""
    # Pretty print selected metrics
    print("\n=== Evaluasi Model ===")
    for label in ["negative", "neutral", "positive", "accuracy"]:
        if label == "accuracy":
            acc = report_dict.get("accuracy", None)
            if acc is not None:
                print(f"Akurasi: {acc:.4f}")
            continue
        row = report_dict.get(label, None)
        if row:
            print(
                f"Label={label} | precision={row.get('precision', 0):.4f} | recall={row.get('recall', 0):.4f} | f1={row.get('f1-score', 0):.4f}"
            )


def main():
    """Fungsi utama: parsing argumen, load data, preprocessing, training/inferensi, simpan & plot."""
    parser = argparse.ArgumentParser(
        description="Analisis sentimen layanan kampus (Indonesia) dari dataset tweet/ulasan CSV."
    )
    parser.add_argument("--csv", required=True, help="Path ke file CSV, misal: unib.csv")
    parser.add_argument(
        "--text-col",
        default="content",
        help="Nama kolom teks pada CSV (default: content; alternatif umum: text)",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Nama kolom label sentimen jika tersedia (misal: label). Jika kosong, gunakan model pra-latih.",
    )
    parser.add_argument(
        "--model",
        choices=["lr", "nb", "bert"],
        default="lr",
        help="Pilih model: lr=LogisticRegression, nb=MultinomialNB, bert=IndoBERT pra-latih (jika tidak ada label)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Ukuran test set (default: 0.2)")
    parser.add_argument(
        "--output",
        default="hasil_sentimen.csv",
        help="Nama file keluaran untuk menyimpan hasil klasifikasi",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"File CSV tidak ditemukan: {args.csv}")
        sys.exit(1)

    # 1) Muat dataset CSV
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        alt = "text" if "text" in df.columns else None
        if alt:
            print(f"Kolom {args.text_col} tidak ditemukan. Menggunakan kolom alternatif: {alt}")
            args.text_col = alt
        else:
            print(f"Kolom teks '{args.text_col}' tidak ditemukan pada CSV. Kolom tersedia: {list(df.columns)}")
            sys.exit(1)

    # 2) Preprocessing teks
    texts_raw = df[args.text_col]
    print("Melakukan preprocessing teks (cleaning, tokenisasi, stopword removal, stemming)...")
    texts_proc = preprocess_series(texts_raw)

    # If label column exists and user didn't select BERT, prefer ML training and evaluation
    # 3) Jika label tersedia dan model klasik dipilih → lakukan training & evaluasi
    if args.label_col and args.label_col in df.columns and args.model in {"lr", "nb"}:
        labels_raw = df[args.label_col]
        labels_norm, _ = ensure_label_column(labels_raw)
        mask = labels_norm.notna() & texts_proc.notna() & texts_proc.str.len().gt(0)
        texts_use = texts_proc[mask]
        labels_use = labels_norm[mask]

        if labels_use.nunique() < 2:
            print("Label kurang dari 2 kelas setelah normalisasi. Beralih ke mode prediksi tanpa label (BERT jika tersedia).")
        else:
            model, vectorizer, y_test, y_pred, report_dict, cm = train_and_evaluate_ml(
                texts_use, labels_use, model_name=args.model, test_size=args.test_size
            )
            print_classification_report(report_dict)
            # Confusion matrix heatmap
            plt.figure(figsize=(5, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["negative", "neutral", "positive"],
                yticklabels=["negative", "neutral", "positive"],
            )
            plt.title("Confusion Matrix")
            plt.xlabel("Prediksi")
            plt.ylabel("Aktual")
            plt.tight_layout()
            plt.show()

            # 4) Fit pada seluruh data untuk menghasilkan prediksi ke semua baris dan menyimpan hasil
            vec_all = vectorizer.fit_transform(texts_proc)
            model.fit(vec_all, labels_use.reindex(texts_proc.index, fill_value=labels_use.mode().iloc[0]))
            preds_all = model.predict(vec_all)

            df_out = df.copy()
            df_out["sentimen"] = preds_all
            df_out.to_csv(args.output, index=False)
            print(f"Hasil klasifikasi disimpan ke: {args.output}")

            # Distribusi sentimen
            plot_distribution(pd.Series(preds_all), title="Distribusi Sentimen (Prediksi)", output_path="distribusi_sentimen.png")
            return

    # 5) Jika tidak ada label/ tidak training → gunakan model pra-latih (jika tersedia)
    print("Tidak menggunakan pelatihan terawasi. Mencoba memuat model pra-latih (IndoBERT/Roberta)...")
    pipe = try_load_indobert_pipeline()
    if pipe is None:
        print("Model pra-latih tidak tersedia. Silakan sediakan kolom label dan gunakan --model lr/nb untuk pelatihan.")
        sys.exit(1)

    preds = predict_with_bert(pipe, texts_proc.fillna("").tolist())
    df_out = df.copy()
    df_out["sentimen"] = preds
    df_out.to_csv(args.output, index=False)
    print(f"Hasil klasifikasi disimpan ke: {args.output}")

    # Tampilkan contoh prediksi
    print("\nContoh 5 prediksi pertama:")
    for i, (t, p) in enumerate(zip(df_out[args.text_col].head(5), df_out["sentimen"].head(5))):
        print(f"{i+1}. [{p}] {t}")

    plot_distribution(pd.Series(preds), title="Distribusi Sentimen (Pra-latih)", output_path="distribusi_sentimen.png")


if __name__ == "__main__":
    main()


