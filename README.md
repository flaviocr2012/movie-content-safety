# 🎬 Movie Content Safety Classifier

> AI-powered system that determines if a movie is appropriate for children aged 5-10

Built with **LangChain**, **FAISS**, and **Groq** — a complete RAG system with an AI Agent for complex queries.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **RAG Classification** | Semantic search to retrieve relevant safety rules |
| **AI Agent** | Answers complex questions using 3 tools |
| **Interactive CLI** | Chat-style interface for movie safety queries |
| **Batch Processing** | Classify multiple movies at once |
| **Movie Database** | 30+ movies with details |
| **Knowledge Base** | 80+ safety Q&A pairs |

---

## 🏗️ Architecture

<div style="font-family: sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-top: 20px; background-color: #f9f9f9;">
  <h3 style="border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-top: 0;">Architecture</h3>

  <!-- User Interface Layer -->
  <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; border: 1px solid #ccc; border-radius: 5px; margin-bottom: 15px; background: #fff;">
    <strong>USER INTERFACE</strong>
    <div>
      <span style="margin-right: 10px;">[ Interactive CLI ]</span>
      <span style="margin-right: 10px;">[ Batch Mode ]</span>
      <span>[ Agent ]</span>
    </div>
  </div>

  <!-- RAG Chain Layer -->
  <div style="display: flex; justify-content: space-around; align-items: center; padding: 15px; border: 1px solid #ccc; border-radius: 5px; margin-bottom: 15px; background: #fff;">
    <div style="text-align: center;"><strong>Retriever</strong><br><span style="font-size: 0.8em; color: #666;">(FAISS)</span></div>
    <span style="font-size: 20px;">&rarr;</span>
    <div style="text-align: center;"><strong>Context</strong><br><span style="font-size: 0.8em; color: #666;">(Top-5 Q&A)</span></div>
    <span style="font-size: 20px;">&rarr;</span>
    <div style="text-align: center;"><strong>LLM</strong><br><span style="font-size: 0.8em; color: #666;">(Groq)</span></div>
  </div>

  <!-- Data Layer -->
  <div style="display: flex; justify-content: space-between; padding: 15px; border: 1px solid #ccc; border-radius: 5px; background: #fff;">
    <div style="border-right: 1px solid #ddd; padding-right: 20px;">
      <strong>Knowledge Base</strong><br>
      <span style="font-size: 0.8em; color: #666;">(80+ Q&A pairs)</span>
    </div>
    <div>
      <strong>Movie Database (CSV)</strong><br>
      <span style="font-size: 0.8em; color: #666;">(30 movies with details)</span>
    </div>
  </div>

</div>

User → RAG Chain (Retriever → Context → LLM) → Data Layer (KB + Movies)


**Flow:** Query → Embedding → FAISS Search → Top-5 Q&A → LLM → Classification

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Groq API key (free) — [Get it here](https://console.groq.com)

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/movie-content-safety.git
cd movie-content-safety
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

> **Note:** Get your free Groq API key from [console.groq.com](https://console.groq.com)

## 🚀 Run

### 1. Build the Vector Index

```bash
python src/vector_store.py
```

### 2. Interactive Movie Classifier

```bash
python src/main.py
```

**Example Interaction:**

```
🎬 Enter movie title (or command): The Lion King
📖 Found 'The Lion King' in CSV database!
📝 Year: 1994 | Rating: 8.5 | Genres: Animation, Adventure, Drama

📌 Result:
Classification: Safe for children
Explanation: The movie is animated, family-friendly, and contains no adult content.
```

### 3. Batch Classification

```bash
python src/main.py --batch --limit 5
```

### 4. AI Agent for Complex Queries

```bash
python src/agent.py
```

**Example Questions:**

- *"Can you find me a movie like The Lion King that is appropriate for a 5-year-old?"*
- *"Is Jurassic Park safe for children?"*
- *"What are the best family movies in the database?"*

## 📊 Sample Outputs

### Agent Response

```
💭 Your question: What movies are safe for children?

📌 Response:
Here are some movies that are safe for children aged 5-10:

| Movie | Rating | Why it's safe |
|-------|--------|---------------|
| Toy Story | G | Animated, no violence, positive messages |
| Finding Nemo | G | Light-hearted adventure, no scary content |
| Frozen | PG | Positive themes of sisterhood and self-acceptance |
| The Lion King | G | Strong moral lessons, no graphic violence |
```

### Movie Classification

```
🎬 Enter movie title (or command): The Conjuring
📖 Found 'The Conjuring' in CSV database!
📝 Year: 2013 | Rating: 7.5 | Genres: Horror, Mystery, Thriller

📌 Result:
Classification: Not safe for children
Explanation: The Conjuring is a horror film with a rating of 7.5 (R). It contains supernatural terror, frightening imagery, and intense suspense typical of the horror genre, making it unsuitable for children aged 5-10.
``` 

## 📁 Project Structure

```
movie-content-safety/
├── data/
│   ├── knowledge_base.csv      # 80+ Q&A safety rules
│   ├── imdb_movies.csv         # 30 movies with details
│   └── faiss_index/            # FAISS vector index
│       ├── index.faiss
│       └── index.pkl
├── src/
│   ├── config.py               # Configuration & API keys
│   ├── vector_store.py         # Build FAISS index
│   ├── rag_chain.py            # RAG pipeline with Groq
│   ├── main.py                 # Interactive CLI
│   └── agent.py                # AI Agent for complex queries
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
└── README.md                   # This file
``` 

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **LangChain** | LLM orchestration framework |
| **FAISS** | Vector similarity search |
| **Groq** | Fast, free LLM inference |
| **HuggingFace** | Embedding models (all-MiniLM-L6-v2) |
| **Python** | Core programming language |
| **Pandas** | CSV data handling |  

## 📦 Dependencies

```txt
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
langchain-groq>=0.1.0
langchain-huggingface
faiss-cpu
sentence-transformers
pandas
python-dotenv
``` 

## 🔮 Next Steps

- [ ] Add web interface (Gradio/Streamlit)
- [ ] Add evaluation framework (evals)
- [ ] Deploy to production
- [ ] Expand knowledge base to 500+ movies
- [ ] Add user feedback loop 

## 📫 Let's Connect!

If you found this project interesting, feel free to reach out!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/flavio-rodrigues-7563b631/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/flaviocr2012)

---
