# Sistema RAG Multi-Provedor no Databricks

## 📋 Visão Geral

Sistema de **Retrieval-Augmented Generation (RAG)** desenvolvido para Databricks que permite fazer perguntas sobre documentos usando múltiplos provedores de IA. O sistema processa documentos, gera embeddings vetoriais e oferece respostas contextualizadas usando OpenAI, Anthropic (Claude), Google Gemini ou Deepseek.

## 🎯 Principais Funcionalidades

- **Processamento Multi-formato**: Suporta PDF, DOCX e PPTX
- **Chunking Inteligente**: Quebra documentos em chunks com overlap configurável
- **Embeddings Multilíngues**: Usa modelo BAAI/bge-m3 otimizado para português
- **Busca Vetorial**: FAISS para recuperação eficiente de contexto relevante
- **Multi-LLM**: Suporta 4 provedores de IA diferentes
- **Persistência Delta Lake**: Armazena documentos, chunks, embeddings e auditoria
- **Auditoria Completa**: Registra todas as interações para análise e melhoria

## 🏗️ Arquitetura

```
Documentos (PDF/DOCX/PPTX)
    ↓
Extração de Texto
    ↓
Chunking (800 chars, overlap 120)
    ↓
Embeddings (BAAI/bge-m3)
    ↓
FAISS Index
    ↓
RAG Pipeline → LLM (OpenAI/Claude/Gemini/Deepseek)
    ↓
Resposta + Auditoria
```

## 📊 Estrutura de Dados (Unity Catalog)

### Tabelas Delta

1. **`bronze.default.docs_raw`**
   - Documentos brutos processados
   - Schema: `doc_id`, `file_name`, `file_type`, `text_content`, `char_count`, `processed_at`

2. **`bronze.default.docs_chunks`**
   - Chunks extraídos dos documentos
   - Schema: `chunk_id`, `doc_id`, `chunk_index`, `chunk_text`, `char_count`

3. **`bronze.default.docs_embeddings`**
   - Vetores de embedding
   - Schema: `chunk_id`, `doc_id`, `embedding` (array de floats)

4. **`bronze.default.rag_audit`**
   - Log de todas as consultas
   - Schema: `query_id`, `timestamp`, `provider`, `model`, `question`, `top_k`, `latency_ms`, `chunks_used`, `answer`

## 🚀 Como Usar

### 1. Configuração de Secrets (Databricks)

```bash
# Configure as secrets no Databricks:
# Scope: OPENAI, Key: OPENAI_API_KEY
# Scope: CLAUDE, Key: ANTHROPIC_API_KEY
# Scope: GEMINI, Key: GEMINI_API_KEY
# Scope: DEEPSEEK, Key: DEEPSEEK_API_KEY
```

### 2. Upload do Documento

Coloque seu documento em:
```
/Volumes/bronze/default/documentos_agent/seu_documento.pdf
```

### 3. Execução do Pipeline

Execute as células do notebook na ordem:

1. **Instalação de dependências**
2. **Restart do Python**
3. **Configurações e imports**
4. **Processamento do documento**
5. **Geração de embeddings**
6. **Interface conversacional**

### 4. Interação

```python
# O sistema perguntará:
# - Provedor (openai/anthropic/gemini/deepseek)
# - Modelo (opcional)
# - Temperatura (0-1)

# Depois, faça suas perguntas:
Você: Qual o tamanho mínimo do recinto para tigres?
🤖 IA: [Resposta contextualizada com citações]
```

## 🔧 Parâmetros Configuráveis

```python
# Chunking
CHUNK_SIZE = 800          # Tamanho do chunk em caracteres
CHUNK_OVERLAP = 120       # Overlap entre chunks

# Busca
TOP_K_DEFAULT = 6         # Número de chunks recuperados

# Embeddings
EMBED_MODEL_NAME = "BAAI/bge-m3"  # Modelo multilíngue
```

## 📦 Dependências

```
pypdf
python-docx
python-pptx
sentence-transformers
faiss-cpu
tqdm
google-generativeai
anthropic
openai
pyspark
numpy
```

## 🎓 Casos de Uso

- **Base de Conhecimento Corporativa**: Consulte manuais, políticas e procedimentos
- **Análise de Documentos Técnicos**: Extraia insights de relatórios e estudos
- **Suporte ao Cliente**: Responda perguntas baseadas em documentação de produtos
- **Conformidade**: Acesso rápido a normas e regulamentações

## 📈 Melhorias Futuras

- [ ] Suporte a mais formatos (Excel, TXT, Markdown)
- [ ] Reranking com cross-encoder
- [ ] Streaming de respostas
- [ ] Interface web (Streamlit/Gradio)
- [ ] Avaliação automática de qualidade (RAGAS)
- [ ] Cache de embeddings para evitar reprocessamento
- [ ] Suporte a múltiplos documentos simultâneos

## 🔒 Segurança

- Secrets gerenciadas pelo Databricks
- Dados persistidos no Unity Catalog com controle de acesso
- Auditoria completa de todas as consultas
- Sem exposição de chaves de API no código

## 📚 Documentação Adicional

- [Documentação para Engenheiros de Dados](DOCS_DATA_ENGINEER.md)
- [Documentação para Engenheiros de IA](DOCS_AI_ENGINEER.md)

## 🤝 Contribuindo

Para melhorias no sistema:
1. Ajuste os parâmetros de chunking para seu caso de uso
2. Experimente diferentes modelos de embedding
3. Teste diferentes provedores de LLM
4. Analise a tabela de auditoria para otimizações

## 📄 Licença

MIT

---

**Desenvolvido para Databricks com Unity Catalog e Delta Lake**