# 🧠 AI CRM MCP Server

Servidor MCP (Model Context Protocol) para gerenciamento de usuários com suporte a busca semântica utilizando embeddings e FAISS.

---

## 🚀 Funcionalidades

- Criação de usuários com persistência em SQLite  
- Geração de embeddings com Sentence Transformers  
- Busca semântica com FAISS  
- Recuperação de usuário por ID  
- Listagem de usuários  
- Testes automatizados  

---

## 🏗️ Arquitetura

- Server (MCP Tools)
- Service Layer
- Embedding Service
- Vector Store (FAISS)
- Database (SQLite)

---

## ⚙️ Setup

### Criar ambiente virtual
python -m venv .venv

### Ativar ambiente
.venv\Scripts\activate

### Instalar dependências
pip install -r requirements.txt

### Rodar servidor
python app/server.py

---

## 🧪 Testes
pytest tests/ -v

---

## 🐳 Docker
docker build -t ai-crm-mcp .
docker run ai-crm-mcp

---

## 👨‍💻 Autor
Danilo Barreto
