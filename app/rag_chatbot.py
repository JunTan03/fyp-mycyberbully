import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from chromadb import PersistentClient

# ✅ Tiny model: phi-2
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"  # will use your GTX 1080 if available
)

# ✅ Embeddings and ChromaDB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = PersistentClient(path="rag_storage")
collection = client.get_or_create_collection(name="tweets_rag")

def rag_chatbot_query(question: str, top_k=5):
    # 🔎 Embed question
    question_embedding = embedding_model.encode(question).tolist()

    # 🔍 Semantic retrieval from Chroma
    results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
    retrieved_docs = results["documents"][0]

    print("🔍 Retrieved from Chroma:", retrieved_docs)

    if not retrieved_docs:
        return "I couldn't find relevant information. Try asking differently."

    context = "\n".join(retrieved_docs)
    prompt = f"""Context:
{context}

Question: {question}
Answer:"""

    try:
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

        output = llm_model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        decoded = llm_tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("Answer:")[-1].strip()

        return answer if answer else "I couldn't generate a helpful answer."
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return "Sorry, something went wrong while generating the response."
