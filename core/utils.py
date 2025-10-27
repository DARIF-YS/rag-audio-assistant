import tempfile
import subprocess
from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from a video and return the temporary audio file path."""
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    command = [
        "ffmpeg", "-i", video_path,
        "-vn",
        "-acodec", "mp3",
        "-y", temp_audio.name
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return temp_audio.name


def transcribe_audio(whisper_model, file_bytes: bytes, is_video: bool = False) -> str:
    """Transcribe an audio or video file using Whisper."""
    suffix = ".mp4" if is_video else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    if is_video:
        temp_path = extract_audio_from_video(temp_path)

    result = whisper_model.transcribe(temp_path)
    return result["text"]


def index_transcription(vector_store, transcription_text: str):
    """Split transcription into chunks and index them in the vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.create_documents([transcription_text])
    ids = [str(uuid4()) for _ in range(len(chunks))]
    vector_store.add_documents(chunks, ids=ids)


def generate_answer(vector_store, llm, question: str) -> str:
    """Generate an answer based on the indexed transcription."""
    template = """
    You are an intelligent assistant specialized in analyzing and understanding audio and video content.
    Answer **only** using the information provided in the context below.
    Do not guess or provide information outside the context. Be precise and concise.

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {
            "context": vector_store.as_retriever(search_kwargs={"k": 3}),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)
