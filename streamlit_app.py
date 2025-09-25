import io
import os
from typing import Optional

os.environ["MPLBACKEND"] = "agg"
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Reuse functions from the CLI script
from sentiment_analysis import (
    preprocess_series,
    ensure_label_column,
    train_and_evaluate_ml,
    try_load_indobert_pipeline,
    predict_with_bert,
)

st.set_page_config(page_title="Analisis Sentimen Layanan Kampus", layout="wide")
st.title("🎓 Analisis Sentimen Layanan Kampus (Indonesia)")
st.caption("Unggah CSV tweet/ulasan Anda, pilih kolom, dan jalankan klasifikasi sentimen.")

# ✅ fallback supaya support streamlit lama maupun baru
# Menggunakan st.cache_data (untuk versi baru) atau st.cache (untuk versi lama)
cache_func = getattr(st, "cache_data", st.cache)

@cache_func(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def render_cm(cm: np.ndarray):
    """Render confusion matrix with better styling"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Neutral", "Positive"],
        yticklabels=["Negative", "Neutral", "Positive"],
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
    ax.set_xlabel("Prediksi", fontsize=12)
    ax.set_ylabel("Aktual", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

def render_distribution(labels: pd.Series, title: str):
    """Render sentiment distribution with better styling"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    counts = labels.value_counts()
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]  # Red, Orange, Green
    
    bars = ax.bar(
        ["Negative", "Neutral", "Positive"], 
        [counts.get("negative", 0), counts.get("neutral", 0), counts.get("positive", 0)],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Sentimen", fontsize=12)
    ax.set_ylabel("Jumlah", fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ==============================================
# Session state setup
# ==============================================
if "df" not in st.session_state:
    st.session_state.df = None
if "text_col" not in st.session_state:
    st.session_state.text_col = None
if "label_col" not in st.session_state:
    st.session_state.label_col = "(tanpa label)"
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "lr"
if "test_size" not in st.session_state:
    st.session_state.test_size = 0.2

# ==============================================
# File upload section
# ==============================================
st.subheader("📁 Upload Dataset")
uploaded = st.file_uploader(
    "Pilih file CSV yang berisi data tweet/ulasan tentang layanan kampus",
    type=["csv"],
    help="File CSV harus mengandung kolom teks untuk dianalisis"
)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
with col_btn1:
    sample_btn = st.button("📊 Gunakan dataset dummy", help="Gunakan file unib.csv jika tersedia")
with col_btn2:
    clear_btn = st.button("🗑️ Bersihkan dataset", help="Hapus dataset dari memori")

if uploaded is not None:
    try:
        with st.spinner("Memuat CSV..."):
            st.session_state.df = load_csv(uploaded)
        st.success(f"✅ CSV berhasil dimuat! Jumlah baris: {len(st.session_state.df)}")
    except Exception as e:
        st.error(f"❌ Gagal membaca CSV: {e}")
elif sample_btn:
    dummy_path = os.path.join(os.getcwd(), "unib.csv")
    if os.path.exists(dummy_path):
        try:
            st.session_state.df = pd.read_csv(dummy_path)
            st.info(f"📊 Menggunakan dataset dummy: unib.csv ({len(st.session_state.df)} baris)")
        except Exception as e:
            st.error(f"❌ Gagal membaca file dummy: {e}")
    else:
        st.error("❌ File dummy unib.csv tidak ditemukan di folder proyek.")
elif clear_btn:
    st.session_state.df = None
    st.session_state.text_col = None
    st.session_state.label_col = "(tanpa label)"
    st.session_state.model_choice = "lr"
    st.session_state.test_size = 0.2
    st.info("🗑️ Dataset dibersihkan dari memori.")

df: Optional[pd.DataFrame] = st.session_state.df

# ==============================================
# Main workflow
# ==============================================
if df is not None:
    st.subheader("👀 Pratinjau Data")
    
    # Show dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Baris", len(df))
    with col2:
        st.metric("Jumlah Kolom", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Show column types
    with st.expander("ℹ️ Informasi Kolom"):
        col_info = pd.DataFrame({
            'Kolom': df.columns,
            'Tipe Data': df.dtypes,
            'Contoh Data': [str(df[col].iloc[0]) if len(df) > 0 else "N/A" for col in df.columns]
        }).astype(str) # Solusi untuk NumpyDtypeException
        st.dataframe(col_info)
        
    st.dataframe(df.head(10)) # Solusi untuk 'use_container_width'

    st.subheader("⚙️ Konfigurasi Analisis")
    with st.form("pengaturan"):
        col1, col2 = st.columns(2)
        
        with col1:
            text_col_options = list(df.columns)
            text_default = 0
            if "content" in df.columns:
                text_default = text_col_options.index("content")
            elif "text" in df.columns:
                text_default = text_col_options.index("text")
            elif "tweet" in df.columns:
                text_default = text_col_options.index("tweet")
                
            text_col = st.selectbox(
                "📝 Pilih kolom teks",
                options=text_col_options,
                index=text_default,
                help="Kolom yang berisi teks untuk dianalisis sentimennya"
            )
            
            label_options = ["(tanpa label)"] + list(df.columns)
            label_default = 0
            if "label" in df.columns:
                label_default = label_options.index("label")
            elif "sentiment" in df.columns:
                label_default = label_options.index("sentiment")
                
            label_col = st.selectbox(
                "🏷️ Pilih kolom label (opsional)",
                options=label_options,
                index=label_default,
                help="Kolom berisi label sentimen untuk mode training (opsional)"
            )
        
        with col2:
            model_choice = st.selectbox(
                "🤖 Pilih model",
                options=["lr", "nb", "bert"],
                index=["lr", "nb", "bert"].index(st.session_state.model_choice),
                help="lr=Logistic Regression, nb=Naive Bayes, bert=IndoBERT pra-latih",
                format_func=lambda x: {
                    "lr": "Logistic Regression",
                    "nb": "Naive Bayes", 
                    "bert": "IndoBERT (Pra-latih)"
                }[x]
            )
            
            test_size = st.slider(
                "📊 Ukuran test set",
                min_value=0.1,
                max_value=0.4,
                value=st.session_state.test_size,
                step=0.05,
                help="Proporsi data untuk testing (hanya untuk mode training)"
            )

        run = st.form_submit_button("🚀 Jalankan Analisis") # Solusi untuk 'use_container_width'

    st.session_state.text_col = text_col
    st.session_state.label_col = label_col
    st.session_state.model_choice = model_choice
    st.session_state.test_size = test_size

    if run:
        if text_col not in df.columns:
            st.error("❌ Kolom teks tidak ditemukan.")
            st.stop()

        # Solusi: Pastikan data mentah diubah menjadi string sebelum diproses
        texts_raw_str = df[text_col].astype(str)
        
        with st.spinner("🔄 Preprocessing teks (cleaning, tokenisasi, stopword removal, stemming)..."):
            texts_proc = preprocess_series(texts_raw_str)
        
        # Solusi untuk RuntimeError: Pastikan tidak ada nilai kosong di teks
        valid_texts = texts_proc.dropna()
        if valid_texts.empty:
            st.warning("Tidak ada teks valid untuk dianalisis.")
            st.stop()

        st.info(f"✅ Preprocessing selesai. {len(valid_texts)}/{len(texts_proc)} teks valid untuk analisis.")

        use_label = label_col != "(tanpa label)" and label_col in df.columns and model_choice in {"lr", "nb"}

        if use_label:
            labels_norm, _ = ensure_label_column(df[label_col])
            mask = labels_norm.notna() & texts_proc.notna() & texts_proc.str.len().gt(0)
            texts_use = texts_proc[mask]
            labels_use = labels_norm[mask]

            if labels_use.nunique() < 2:
                st.warning("⚠️ Label kurang dari 2 kelas setelah normalisasi. Beralih ke inferensi tanpa label.")
                use_label = False
            else:
                st.info(f"📊 Menggunakan {len(texts_use)} sampel dengan label valid.")

        if use_label:
            st.subheader("🎯 Mode Training & Evaluasi")
            
            with st.spinner("🔄 Melatih dan mengevaluasi model..."):
                model, vectorizer, y_test, y_pred, report_dict, cm = train_and_evaluate_ml(
                    texts_use, labels_use, model_name=model_choice, test_size=test_size
                )

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Metrik Evaluasi")
                acc = report_dict.get("accuracy", None)
                if acc is not None:
                    st.metric("Akurasi", f"{acc:.4f}", help="Proporsi prediksi yang benar")

                metrics_data = []
                for label in ["negative", "neutral", "positive"]:
                    if label in report_dict:
                        metrics_data.append({
                            "Sentimen": label.capitalize(),
                            "Precision": f"{report_dict[label].get('precision', 0):.4f}",
                            "Recall": f"{report_dict[label].get('recall', 0):.4f}",
                            "F1-Score": f"{report_dict[label].get('f1-score', 0):.4f}",
                            "Support": int(report_dict[label].get('support', 0))
                        })
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df) # Solusi untuk 'use_container_width'
            
            with col2:
                st.subheader("🔥 Confusion Matrix")
                render_cm(cm)

            with st.spinner("🔄 Membuat prediksi final..."):
                vec_all = vectorizer.fit_transform(texts_proc)
                model.fit(vec_all, labels_use.reindex(texts_proc.index, fill_value=labels_use.mode().iloc[0]))
                preds_all = model.predict(vec_all)

            df_out = df.copy()
            df_out["sentimen"] = preds_all
            
            st.subheader("📊 Distribusi Sentimen")
            render_distribution(pd.Series(preds_all), title="Distribusi Sentimen (Hasil Prediksi)")

        else:
            st.subheader("🤖 Mode Inferensi (Model Pra-latih)")
            
            with st.spinner("🔄 Memuat model IndoBERT dan melakukan inferensi..."):
                pipe = try_load_indobert_pipeline()
                if pipe is None:
                    st.error("❌ Model pra-latih tidak tersedia. Instal transformers/torch atau gunakan mode training lr/nb.")
                    st.stop()
                
                # Solusi untuk RuntimeError: Pastikan hanya teks yang valid yang diolah
                preds = predict_with_bert(pipe, valid_texts.tolist())

            df_out = df.copy()
            df_out["sentimen"] = preds
            
            st.subheader("📊 Distribusi Sentimen")
            render_distribution(pd.Series(preds), title="Distribusi Sentimen (Model Pra-latih)")

        # Download section
        st.subheader("💾 Download Hasil")
        csv_buf = io.StringIO()
        df_out.to_csv(csv_buf, index=False)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                "📁 Download CSV",
                data=csv_buf.getvalue(),
                file_name="hasil_sentimen.csv",
                mime="text/csv",
                help="Download hasil analisis dalam format CSV"
            )
        with col2:
            st.info(f"File berisi {len(df_out)} baris dengan kolom sentimen tambahan")

        # Preview results
        st.subheader("👁️ Contoh Hasil Prediksi")
        preview_df = df_out[[text_col, "sentimen"]].head(10).copy()
        preview_df.columns = ["Teks", "Sentimen"]
        
        emoji_map = {"negative": "😞", "neutral": "😐", "positive": "😊"}
        preview_df["Sentimen"] = preview_df["Sentimen"].apply(lambda x: f"{emoji_map.get(x, '❓')} {x}")
        
        st.dataframe(preview_df) # Solusi untuk 'use_container_width'

else:
    st.subheader("📋 Cara Penggunaan")
    st.markdown("""
    1. **Upload file CSV** atau gunakan dataset dummy
    2. **Pilih kolom teks** yang akan dianalisis
    3. **Pilih kolom label** (opsional, untuk mode training)
    4. **Pilih model** yang akan digunakan:
        - **Logistic Regression**
        - **Naive Bayes**
        - **IndoBERT**
    5. **Atur ukuran test set**
    6. **Jalankan analisis** dan lihat hasilnya!
    """)