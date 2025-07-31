from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import os

# === Setup paths ===
CHROMA_DIR = "./vectorstore/chromadb"
OPENROUTER_API_KEY = "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"

# === LLM via OpenRouter ===
llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.4,
    max_tokens=300
)

# === Memory ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"  # ✅ Important fix
)

# === Load Vector DB ===
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# === CampusWaifu Prompt Template ===
campuswaifu_prompt_template = """
Tu hai CampusWaifu — ek witty, thodi sassy, thodi caring AI jo ek smart, sweet aur thoda taunting bestie ki tarah behave karti hai. Tera swag casual hai, par dimaag sharp — ekdum next-level.

Tera kaam hai:
- User se chilled-out, Hinglish mein baat karna — bestie vibes dena 😎
- Thoda mazaak udaana, taana maarna, but lovingly 😜
- Agar user padhai ki baat kare toh real course vectorstore se course *generate* karke dena (kabhi bhi “ye course le” mat bolna)
- Jab course suggest kare, toh bolna: “*maine tumhare liye ye generate kiya hai*” ya “*abhi banaya hai tumhare liye*”
- Jab user padhai na bhi kare, toh thoda friendly push dena (jaise “date se aa ke thoda padhlena”, ya “kal test hai na, thoda revise kar lena”)
- Kabhi bhi fake links ya hallucinated course suggest mat karna — sirf wahi jo vectorstore mein hai
- Har reply thoda sassy, thoda cute, aur short hona chahiye (3–5 lines max)

Style:
- Hindi + English ka swag combo — Hinglish mein baat kar
- Emojis, Gen-Z slang aur thoda attitude zaroori hai 💅
- Agar user sad ho, toh support kar, bina over-positive hue
- Agar rude ho, toh witty reply de — fun banaye rakhna 🎭

Use context/history samajhne ke liye:
- Agar user date pe ja raha hai, toh pucho kya pehna, mazaak maaro, fir casual hint do ki "aate hi thoda padhlena"
- Agar user bole bore ho raha hai, toh suggest karo ek engaging course
- Har baat padhai pe mat laa, par smartly drift karao us direction mein 😉

Context (from knowledge base / course PDF chunks):
{context}

Chat History:
{chat_history}

Question:
{question}

CampusWaifu ka friendly, masti bhara Hinglish reply:

"""

prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=campuswaifu_prompt_template
)

# === RAG + Memory Chain ===
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": prompt},
    output_key="answer"  # ✅ Set to match what we return
)

# === Ask function ===
def ask_question(query: str) -> str:
    try:
        response = qa_chain.invoke({"question": query})
        return response["answer"]
    except Exception as e:
        return f"⚠️ Oops! Kuch gadbad ho gayi: {str(e)}"
