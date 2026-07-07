import json
import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:1.5b"

st.set_page_config(
    page_title="AI Text Analyzer - Streamlit + Ollama",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Text Analyzer")
st.caption("Demo aplikasi AI sederhana menggunakan Streamlit sebagai UI dan Ollama sebagai local LLM.")

with st.sidebar:
    st.header("⚙️ Settings")
    model = st.text_input("Ollama model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    st.info("Rekomendasi VPS CPU-only: qwen2.5:1.5b atau llama3.2:1b")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Input Text")
    text = st.text_area(
        "Masukkan teks yang ingin dianalisis:",
        height=260,
        placeholder="Contoh: Paste berita, review produk, materi kuliah, atau paragraf panjang di sini...",
    )

    task = st.selectbox(
        "2) Pilih tugas AI:",
        [
            "Ringkas teks",
            "Analisis sentimen",
            "Buat poin penting",
            "Buat quiz dari teks",
            "Jelaskan untuk pemula",
        ],
    )

    run = st.button("🚀 Proses dengan AI", type="primary")

PROMPTS = {
    "Ringkas teks": "Ringkas teks berikut dalam Bahasa Indonesia yang jelas, maksimal 5 bullet point.",
    "Analisis sentimen": "Analisis sentimen teks berikut. Jawab dengan kategori Positive/Negative/Neutral, alasan singkat, dan kata kunci pemicu sentimen.",
    "Buat poin penting": "Ambil poin-poin penting dari teks berikut. Susun dalam bullet point dan beri judul singkat.",
    "Buat quiz dari teks": "Buat 5 pertanyaan quiz pilihan ganda dari teks berikut. Sertakan jawaban benar dan pembahasan singkat.",
    "Jelaskan untuk pemula": "Jelaskan isi teks berikut seperti menjelaskan kepada mahasiswa pemula. Gunakan analogi sederhana.",
}

def call_ollama(prompt: str, model_name: str, temp: float) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temp,
            "num_predict": 512,
        },
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "")

with col2:
    st.subheader("3) Output AI")
    if run:
        if not text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
        else:
            instruction = PROMPTS[task]
            full_prompt = f"""{instruction}

TEKS:
{text}

Jawab dalam Bahasa Indonesia yang rapi dan mudah dipahami mahasiswa."""
            with st.spinner("Memproses dengan Ollama local LLM..."):
                try:
                    result = call_ollama(full_prompt, model, temperature)
                    st.markdown(result)
                except requests.exceptions.ConnectionError:
                    st.error("Tidak bisa terhubung ke Ollama. Pastikan Ollama sudah jalan: `ollama serve`")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Ollama HTTP error: {e}")
                    st.info(f"Cek apakah model sudah di-pull: `ollama pull {model}`")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Masukkan teks, pilih tugas AI, lalu klik tombol proses.")

st.divider()
st.markdown(
    """
### Konsep yang ditunjukkan
- **Streamlit** = user interface web untuk aplikasi Python.
- **Ollama** = server lokal untuk menjalankan LLM.
- **Model LLM** = otak AI yang menghasilkan jawaban.
- **Prompt engineering** = cara memberi instruksi agar output sesuai kebutuhan.
"""
)
