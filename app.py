import streamlit as st
from core.models import load_models
from core.utils import transcribe_audio, index_transcription, generate_answer

st.set_page_config(page_title="Audio/Video QA App", layout="wide", page_icon="🔊")
st.title("🔊 Audio/Video QA App")
st.caption("Transcribe your **audio or video** and ask questions with AI")

st.markdown("""
<style>
    .block-container {
      padding-top: 2.4rem;
      padding-bottom: 1rem;
      padding-left: 1rem;
      padding-right: 1rem;
      max-width: 1200px;
    }
</style>
""", unsafe_allow_html=True)

# Load models
whisper_model, vector_store, llm = load_models()

# Session state initialization
if "media_loaded" not in st.session_state:
    st.session_state["media_loaded"] = False
if "transcription_text" not in st.session_state:
    st.session_state["transcription_text"] = None

media_file = st.file_uploader("Upload an audio or video file", type=["mp3", "wav", "m4a", "mp4", "mov", "avi", "mkv"])

if media_file:
    new_file_id = f"{media_file.name}_{media_file.size}"
    if st.session_state.get("current_file_id") != new_file_id:
        st.session_state.clear()
        st.session_state["current_file_id"] = new_file_id
        st.rerun()
else:
    st.info("Upload an audio or video file to get started.")
    st.stop()

col1, col2 = st.columns([1.2, 1])

with col1:
    file_type = media_file.type
    is_video = file_type.startswith("video")

    if is_video:
        st.video(media_file)
    else:
        st.audio(media_file)

    if not st.session_state["media_loaded"]:
        if st.button("Transcribe and index", use_container_width=True):
            with st.spinner("Transcribing..."):
                transcription = transcribe_audio(whisper_model, media_file.read(), is_video=is_video)
                index_transcription(vector_store, transcription)
                st.session_state["transcription_text"] = transcription
                st.session_state["media_loaded"] = True
            st.success("Transcription completed!")
            st.balloons()

    if st.session_state["media_loaded"]:
        st.text_area("Transcription", st.session_state["transcription_text"], height=400)

with col2:
    if st.session_state["media_loaded"]:
        question = st.text_input("Ask a question about the content:")
        if st.button("Generate answer", use_container_width=True) and question:
            with st.spinner("Generating answer..."):
                answer = generate_answer(vector_store, llm, question)
            st.markdown("#### Answer:")
            st.success(answer)
    else:
        st.info("Transcribe your content before asking a question.")
