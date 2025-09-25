# **AI-Powered FAQ Assistant**

An **AI-powered FAQ chatbot** that answers employee questions related to **HR policies, leave, payroll, and workplace guidelines**.  
This project uses **Retrieval-Augmented Generation (RAG)** with:
- **Hybrid Retrieval** → BM25 (keyword-based) + Semantic Search (embeddings with ChromaDB)
- **LLM Generation** → Hugging Face Mistral-7B-Instruct
- **FastAPI Backend** → for scalable Q&A APIs
- **Streamlit UI** → for a clean, interactive chat interface

---

## **Features**
- **Hybrid Retrieval**: Combines keyword and semantic search for accurate FAQ retrieval.
- **Dynamic Answer Generation**: Uses retrieved FAQ context with an LLM to produce human-like answers.
- **Streaming Responses**: Tokens stream in real-time to the UI for a typing effect.
- **Clean, Modern UI**:
  - Fixed suggestion buttons for quick actions.
  - Left-aligned assistant messages, right-aligned user messages.
- **Session Management**: Maintains persistent history per chat session.

---

## **Folder Structure**
```bash
AI-POWERED-FAQ-ASSISTANT/
│
├── app/
│ ├── bootstrap.py # Build BM25 index and Chroma vector DB
│ ├── main.py # FastAPI backend
│ └── ui.py # Streamlit frontend
│
├── backend/
│ ├── document_loader.py # Preprocess raw CSV data
│ ├── embedder.py # Embedding generator using SentenceTransformers
│ ├── rag_pipeline.py # Hybrid retrieval + LLM generation pipeline
│ ├── store_index_bm25.py # BM25 index builder
│ └── vector_store.py # ChromaDB vector store manager
│
├── config/
│ ├── prompts.yaml # Prompt template for LLM
│ └── settings.yaml # Configuration (LLM, retriever, etc.)
│
├── data/
│ ├── raw/faqs.csv # Raw FAQ dataset
│ ├── processed/ # Processed CSV after cleaning
│ ├── indexes/ # Saved BM25 index
│ └── vectordb/ # Chroma vector database
│
├── requirements.txt
├── README.md
└── .env
```


---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone <your-repo-url>
cd AI-POWERED-FAQ-ASSISTANT
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv .venv
source .venv/bin/activate      # On Linux/Mac
.venv\Scripts\activate         # On Windows
```

### **3. Install Dependencies**
Make sure you have Python 3.9+ installed, then:
```bash
pip install -r requirements.txt
```

### **4. Environment Variables**

Create a .env file in the project root:

```bash
HUGGINGFACEHUB_API_TOKEN=<your_huggingface_api_token>
```

---

## **Data Preparation**

The assistant uses a preloaded FAQ dataset (data/raw/faqs.csv).
Run the bootstrap script to:

* Clean and preprocess the data.
* Build a BM25 index for keyword-based retrieval.
* Populate ChromaDB with semantic embeddings.

```bash
python app/bootstrap.py
```

---

## **Expected output:**
```bash
Preprocessing raw CSV -> data/processed/processed_faqs.csv
Building BM25 index -> data/indexes/bm25_index.pkl
Populating ChromaDB collection -> hr_policy_qa
=== Bootstrap complete! ===
```

---

## **Run the Application**
1. Start the Backend

Launch the FastAPI backend server:
```bash
uvicorn app.main:app --reload
```
The backend will run at: http://127.0.0.1:8000

2. Launch the UI

Open a second terminal and run:
```bash
streamlit run app/ui.py
```
This will open the Streamlit interface in your default browser.

---

## **Requirements**

Key dependencies included in requirements.txt:
```bash
fastapi

uvicorn

streamlit

chromadb

rank-bm25

sentence-transformers

transformers

python-dotenv

pandas

pyyaml

requests
```
Install them all using:
```bash
pip install -r requirements.txt
```

---

## **API Endpoints**

| Method | Endpoint      | Description                |
| ------ | ------------- | -------------------------- |
| `GET`  | `/`           | Health check               |
| `POST` | `/ask/stream` | Streaming tokenized answer |


---

## **Usage**

Choose from Quick Suggestion buttons or type a custom question.

The backend:

* Retrieves the top matching FAQs (BM25 + semantic search).

* Builds a contextual prompt for the LLM.

* Streams the generated response back to the UI.

* View results in real-time with session history maintained.

---

## **Example Questions**

* "How do I download my payslip?"

* "What is the notice period for resignation?"

* "Are there any educational allowances?"

* "What documents are needed for reimbursement?"

---




## **Future Improvements**

Add multilingual FAQ support.

Use a more advanced reranking model (e.g., cross-encoder).

Add analytics dashboard for query insights.

Deploy on cloud (AWS/GCP/Azure) with Docker.




