# 📖 PDF Q&A Chatbot using Streamlit & Hugging Face API

## 🚀 Overview
This project is a **PDF-based Q&A chatbot** built with **Streamlit**, **FAISS** for similarity search, and **Hugging Face Inference API** for generating answers. Users can upload a **PDF file**, ask questions about its content, and receive answers using an **LLM (e.g., Mistral-7B, Falcon-7B, Llama-3, etc.)**.

---

## 🎯 Features
✅ **Upload PDFs** & extract text + tables 📄  
✅ **Vector search with FAISS** for retrieving relevant text 🔍  
✅ **Generate answers using Hugging Face Inference API** 🤖  
✅ **Streamlit UI for easy interaction** 🖥️  
✅ **Supports any Hugging Face text generation model** 🔄  

---

## 🛠️ Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/ShubhamMandowara/llm_rag.git
cd pdf-qna-chatbot
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🔑 Set Up Hugging Face API Token
1. Get your API token from [Hugging Face Tokens](https://huggingface.co/settings/tokens).
2. When running the Streamlit app, enter the API token in the provided input field.
3. Or, store it in **Streamlit secrets** by adding to `.streamlit/secrets.toml`:
   ```toml

```

