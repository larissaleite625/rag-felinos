# Documentação para Engenheiros de IA - Sistema RAG

## 🎯 Objetivo

Este documento descreve a arquitetura de IA, pipeline RAG, modelos utilizados, estratégias de retrieval e geração do sistema.

---

## 🧠 Arquitetura RAG (Retrieval-Augmented Generation)

### Fluxo Completo

```
┌──────────────┐
│   Pergunta   │
│   do Usuário │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embedding   │
│  da Query    │ ← BAAI/bge-m3
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    FAISS     │
│  Search      │ ← Top-K chunks
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Prompt     │
│ Construction │ ← Contexto + Pergunta
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     LLM      │
│  Generation  │ ← OpenAI/Claude/Gemini/Deepseek
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Resposta   │
│ Contextualizada
└──────────────┘
```

### Componentes-Chave

1. **Chunking**: Divisão semântica do documento
2. **Embedding Model**: Conversão de texto para vetores
3. **Vector Store**: Índice FAISS para busca eficiente
4. **Retriever**: Sistema de recuperação de contexto
5. **LLM**: Geração da resposta final

---

## 📝 Chunking Strategy

### Parâmetros

```python
CHUNK_SIZE = 800        # Caracteres por chunk
CHUNK_OVERLAP = 120     # Overlap entre chunks
```

### Algoritmo

```python
def chunk_text(text: str, chunk_size=800, overlap=120) -> List[str]:
    # 1. Quebra por parágrafos (preserva estrutura semântica)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    
    # 2. Agrupa parágrafos em chunks
    chunks = []
    buf = ""
    
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= chunk_size:
            buf += para + "\n"
        else:
            if buf:
                chunks.append(buf.strip())
            buf = para + "\n"
    
    if buf:
        chunks.append(buf.strip())
    
    # 3. Aplica overlap (mantém contexto entre chunks)
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_tail = chunks[i-1][-overlap:]
            chunk = prev_tail + " " + chunk
        overlapped.append(chunk)
    
    return overlapped
```

### Por que essa estratégia?

**Vantagens:**
- ✅ Preserva estrutura semântica (parágrafos)
- ✅ Overlap evita perda de contexto nas bordas
- ✅ Chunks uniformes (~800 chars) otimizam embeddings
- ✅ Adaptável para diferentes tipos de documentos

**Trade-offs:**
- Chunks muito pequenos: perdem contexto
- Chunks muito grandes: diluem relevância
- 800 chars é sweet spot para BAAI/bge-m3

---

## 🔢 Embedding Model: BAAI/bge-m3

### Características

```python
EMBED_MODEL_NAME = "BAAI/bge-m3"

model = SentenceTransformer(EMBED_MODEL_NAME)
```

**Especificações:**
- **Dimensões**: 1024
- **Max Length**: 8192 tokens
- **Multilingual**: Suporta 100+ idiomas (incluindo PT-BR)
- **Training**: Contrastive learning + in-batch negatives
- **Performance**: SOTA em várias tarefas de retrieval

### Por que BGE-M3?

| Critério | BGE-M3 | Alternativas |
|----------|--------|--------------|
| Português | ⭐⭐⭐⭐⭐ | all-MiniLM (⭐⭐⭐) |
| Velocidade | ⭐⭐⭐⭐ | OpenAI Ada-002 (⭐⭐) |
| Custo | Grátis | OpenAI (pago) |
| Offline | ✅ | OpenAI (❌) |
| Dimensões | 1024 | Ada-002 (1536) |

### Geração de Embeddings

```python
# Batch processing para eficiência
BATCH_SIZE = 32

embeddings = []
for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch = chunks[i:i+BATCH_SIZE]
    batch_embeds = model.encode(
        batch,
        normalize_embeddings=True,  # Normalização L2
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )
    embeddings.extend(batch_embeds)
```

**Normalização L2:**
- Converte vetores para unitários (norma = 1)
- Permite usar Inner Product = Cosine Similarity
- Acelera busca no FAISS

---

## 🔍 Vector Search com FAISS

### Configuração do Índice

```python
dimension = 1024
index = faiss.IndexFlatIP(dimension)  # Inner Product (= Cosine após normalização)

# Adicionar vetores
index.add(embeddings_np)  # numpy array [N, 1024]
```

### Por que IndexFlatIP?

**Vantagens:**
- Busca exata (100% recall)
- Simples e robusto
- Suficiente para até ~1M vetores

**Para escala maior (>1M vetores):**

```python
# Approximate Nearest Neighbor (ANN)
nlist = 100  # Número de clusters
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

# Treinar índice
index.train(embeddings_np)
index.add(embeddings_np)

# Busca (ajustar nprobe para recall vs speed)
index.nprobe = 10  # Procurar em 10 clusters
```

### Retrieval Function

```python
def retrieve(question: str, k: int = 6) -> List[Tuple[str, str, float]]:
    # 1. Gerar embedding da query
    q_embed = model.encode([question], normalize_embeddings=True)
    
    # 2. Buscar top-k no FAISS
    scores, indices = index.search(q_embed, k)
    
    # 3. Recuperar chunks correspondentes
    results = []
    for idx, score in zip(indices[0], scores[0]):
        doc_id = chunks_list[idx]["doc_id"]
        chunk_id = chunks_list[idx]["chunk_id"]
        text = chunks_list[idx]["text"]
        results.append((doc_id, chunk_id, text, score))
    
    return results
```

**Métricas de Similaridade:**
- Score > 0.7: Alta relevância
- Score 0.5-0.7: Média relevância
- Score < 0.5: Baixa relevância

---

## 🎨 Prompt Engineering

### Template RAG

```python
def build_prompt(question: str, passages: List[str]) -> str:
    context = "\n\n".join([
        f"[{i}] {text}" 
        for i, (doc_id, chunk_id, text) in enumerate(passages)
    ])
    
    prompt = f"""Você é um assistente especializado em responder perguntas com base em documentos fornecidos.

CONTEXTO:
{context}

PERGUNTA: {question}

INSTRUÇÕES:
- Responda APENAS com informações do CONTEXTO acima
- Se a resposta não estiver no contexto, diga "Não encontrei essa informação nos documentos"
- Cite as fontes usando [0], [1], etc.
- Seja preciso e objetivo

RESPOSTA:"""
    
    return prompt
```

### Boas Práticas

**✅ DO:**
- Instruções claras e específicas
- Estrutura bem definida (Contexto → Pergunta → Instruções)
- Pedir citações das fontes
- Definir comportamento quando não há resposta

**❌ DON'T:**
- Prompts vagos ou ambíguos
- Misturar contexto e instruções
- Permitir respostas fora do contexto
- Prompts muito longos (> 4000 tokens)

---

## 🤖 Multi-LLM Integration

### Provedores Suportados

```python
PROVIDERS = {
    "openai": {
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    },
    "anthropic": {
        "default_model": "claude-sonnet-4-5-20250929",
        "models": ["claude-opus-4", "claude-sonnet-4-5", "claude-haiku-4"]
    },
    "gemini": {
        "default_model": "gemini-1.5-flash",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"]
    },
    "deepseek": {
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat"]
    }
}
```

### Implementação Unificada

```python
def call_llm(provider: str, prompt: str, model: str = None, temperature: float = 0.0) -> str:
    if provider == "openai":
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    
    elif provider == "anthropic":
        client = anthropic.Anthropic(api_key=ANTH_KEY)
        response = client.messages.create(
            model=model or "claude-sonnet-4-5-20250929",
            max_tokens=2000,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    elif provider == "gemini":
        genai.configure(api_key=GEMINI_KEY)
        model_obj = genai.GenerativeModel(model or "gemini-1.5-flash")
        response = model_obj.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        return response.text
    
    elif provider == "deepseek":
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"},
            json={
                "model": model or "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
        )
        return response.json()["choices"][0]["message"]["content"]
```

### Comparação de Modelos

| Provider | Modelo | Latência | Custo | Qualidade PT-BR | Use Case |
|----------|--------|----------|-------|-----------------|----------|
| OpenAI | gpt-4o-mini | ⚡⚡⚡ | $ | ⭐⭐⭐⭐ | Produção geral |
| OpenAI | gpt-4o | ⚡⚡ | $$$ | ⭐⭐⭐⭐⭐ | Tarefas complexas |
| Anthropic | Claude Sonnet 4.5 | ⚡⚡⚡ | $$ | ⭐⭐⭐⭐⭐ | Melhor em PT-BR |
| Anthropic | Claude Opus 4 | ⚡ | $$$$ | ⭐⭐⭐⭐⭐ | Máxima qualidade |
| Gemini | 1.5-flash | ⚡⚡⚡⚡ | $ | ⭐⭐⭐⭐ | Baixo custo |
| Gemini | 1.5-pro | ⚡⚡ | $$ | ⭐⭐⭐⭐ | Balance |
| Deepseek | deepseek-chat | ⚡⚡⚡ | $ | ⭐⭐⭐ | Alternativa barata |

**Recomendação:**
- **Produção**: Claude Sonnet 4.5 (melhor PT-BR + custo razoável)
- **Desenvolvimento**: GPT-4o-mini (rápido + barato)
- **Máxima Qualidade**: Claude Opus 4 ou GPT-4o

---

## 📊 Hiperparâmetros e Tuning

### Top-K (Número de Chunks)

```python
TOP_K_DEFAULT = 6
```

**Efeito do Top-K:**

| K | Contexto | Latência | Custo | Recall | Precisão |
|---|----------|----------|-------|--------|----------|
| 3 | Pouco | Rápido | Baixo | Baixo | Alto |
| 6 | Adequado | Médio | Médio | Bom | Bom |
| 10 | Muito | Lento | Alto | Alto | Baixo |

**Como escolher:**
- K pequeno (3-4): Perguntas simples, respostas diretas
- K médio (6-8): Default, funciona bem para maioria dos casos
- K grande (10+): Perguntas complexas que precisam de muito contexto

### Temperature

```python
temperature = 0.0  # Default: determinístico
```

**Efeito:**
- **0.0**: Respostas consistentes, factuais
- **0.3-0.5**: Leve variação, ainda factual
- **0.7-1.0**: Criativo, menos consistente

**Recomendação para RAG:**
- Use temperature = 0.0 para respostas factuais
- Use 0.3-0.5 apenas se precisar de variação estilística

### Context Window

**Limites por Modelo:**
- GPT-4o: 128k tokens
- Claude Sonnet 4.5: 200k tokens
- Gemini 1.5: 1M tokens

**Estimativa de Uso:**
```
Prompt template: ~200 tokens
6 chunks × 200 tokens: ~1200 tokens
Total: ~1400 tokens (bem abaixo dos limites)
```

---

## 🎯 Retrieval Optimization

### 1. Reranking (Opcional)

Para melhorar precisão, adicione um cross-encoder:

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retrieve_with_reranking(question: str, k: int = 10) -> List[str]:
    # 1. Retrieval inicial (top-2K)
    candidates = retrieve(question, k=2*k)
    
    # 2. Reranking
    pairs = [[question, text] for _, _, text in candidates]
    scores = reranker.predict(pairs)
    
    # 3. Ordenar por novo score
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    return [item[0] for item in reranked[:k]]
```

### 2. Hybrid Search (Dense + Sparse)

Combine busca vetorial (semântica) com BM25 (léxica):

```python
from rank_bm25 import BM25Okapi

# BM25 para busca por palavras-chave
bm25 = BM25Okapi([chunk.split() for chunk in chunks])

def hybrid_retrieve(question: str, k: int = 6, alpha: float = 0.5) -> List[str]:
    # 1. Busca vetorial
    dense_results = retrieve(question, k=k*2)
    dense_scores = {chunk_id: score for _, chunk_id, _, score in dense_results}
    
    # 2. Busca BM25
    bm25_scores = bm25.get_scores(question.split())
    sparse_scores = {chunk_id: score for chunk_id, score in zip(chunk_ids, bm25_scores)}
    
    # 3. Combinar scores (weighted)
    combined = {}
    for chunk_id in set(dense_scores) | set(sparse_scores):
        d_score = dense_scores.get(chunk_id, 0)
        s_score = sparse_scores.get(chunk_id, 0)
        combined[chunk_id] = alpha * d_score + (1-alpha) * s_score
    
    # 4. Top-K final
    top_chunks = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
    return [chunk_id for chunk_id, _ in top_chunks]
```

### 3. Query Expansion

Expanda a query para melhorar recall:

```python
def expand_query(question: str) -> List[str]:
    # Usar LLM para gerar variações da pergunta
    prompt = f"Gere 3 variações da seguinte pergunta:\n{question}"
    variations = call_llm("openai", prompt).split("\n")
    
    return [question] + variations

def retrieve_with_expansion(question: str, k: int = 6) -> List[str]:
    all_results = []
    for query in expand_query(question):
        results = retrieve(query, k=k//2)
        all_results.extend(results)
    
    # Deduplicate e retornar top-K
    unique = list(set(all_results))
    return unique[:k]
```

---

## 📈 Avaliação e Métricas

### Métricas de Retrieval

```python
def evaluate_retrieval(queries, ground_truth):
    metrics = {
        "recall@k": [],
        "precision@k": [],
        "mrr": []  # Mean Reciprocal Rank
    }
    
    for query, relevant_docs in zip(queries, ground_truth):
        retrieved = retrieve(query, k=10)
        retrieved_ids = [chunk_id for _, chunk_id, _ in retrieved]
        
        # Recall@K
        recall = len(set(retrieved_ids) & set(relevant_docs)) / len(relevant_docs)
        metrics["recall@k"].append(recall)
        
        # Precision@K
        precision = len(set(retrieved_ids) & set(relevant_docs)) / len(retrieved_ids)
        metrics["precision@k"].append(precision)
        
        # MRR
        for i, chunk_id in enumerate(retrieved_ids):
            if chunk_id in relevant_docs:
                metrics["mrr"].append(1 / (i+1))
                break
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

### Métricas de Geração

```python
from rouge import Rouge
from bert_score import score as bert_score

def evaluate_generation(predictions, references):
    rouge = Rouge()
    
    # ROUGE (overlap de n-gramas)
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    
    # BERTScore (similaridade semântica)
    P, R, F1 = bert_score(predictions, references, lang="pt")
    
    return {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "bert-score": F1.mean().item()
    }
```

### RAGAS (RAG Assessment)

Framework especializado para avaliar RAG:

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

results = evaluate(
    dataset,
    metrics=[
        faithfulness,        # Resposta é fiel ao contexto?
        answer_relevancy,    # Resposta é relevante para a pergunta?
        context_precision    # Contexto recuperado é relevante?
    ]
)
```

---

## 🛠️ Troubleshooting

### Problema: Respostas imprecisas

**Diagnóstico:**
```python
# Verificar qualidade do retrieval
results = retrieve("pergunta de teste", k=10)
for i, (doc_id, chunk_id, text, score) in enumerate(results):
    print(f"[{i}] Score: {score:.3f}")
    print(f"    {text[:200]}...\n")
```

**Soluções:**
- Aumentar Top-K se score médio > 0.6
- Melhorar chunking se textos estão cortados
- Considerar reranking se ordem está errada

### Problema: Respostas fora do contexto

**Causa:** Prompt não está restritivo o suficiente

**Solução:**
```python
# Adicionar ao prompt:
"""
IMPORTANTE: Use APENAS as informações do CONTEXTO fornecido.
Se a informação não estiver no contexto, diga claramente:
"Não encontrei essa informação nos documentos fornecidos."
"""
```

### Problema: Latência alta

**Diagnóstico:**
```python
import time

t0 = time.time()
results = retrieve(question, k=6)
t1 = time.time()
answer = call_llm("openai", prompt)
t2 = time.time()

print(f"Retrieval: {(t1-t0)*1000:.0f}ms")
print(f"Generation: {(t2-t1)*1000:.0f}ms")
```

**Soluções:**
- Retrieval lento: Usar FAISS IVF index
- Generation lento: Trocar modelo (GPT-4o → GPT-4o-mini)
- Reduzir Top-K se possível

---

## 🚀 Melhorias Futuras

### 1. Streaming de Respostas

```python
def ask_streaming(question: str):
    passages = retrieve(question)
    prompt = build_prompt(question, passages)
    
    # OpenAI streaming
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

### 2. Conversational RAG (Chat History)

```python
chat_history = []

def ask_with_history(question: str):
    # Incluir histórico no prompt
    history_str = "\n".join([
        f"User: {q}\nAssistant: {a}"
        for q, a in chat_history[-3:]  # Últimas 3 trocas
    ])
    
    prompt = f"""
    HISTÓRICO:
    {history_str}
    
    CONTEXTO: ...
    PERGUNTA ATUAL: {question}
    """
    
    answer = call_llm("openai", prompt)
    chat_history.append((question, answer))
    return answer
```

### 3. Filtros de Metadados

```python
def retrieve_with_filters(question: str, doc_type: str = None, date_range: tuple = None):
    results = retrieve(question, k=20)
    
    # Filtrar por metadados
    filtered = []
    for doc_id, chunk_id, text, score in results:
        metadata = get_metadata(doc_id)
        
        if doc_type and metadata['type'] != doc_type:
            continue
        if date_range and not (date_range[0] <= metadata['date'] <= date_range[1]):
            continue
        
        filtered.append((doc_id, chunk_id, text, score))
    
    return filtered[:6]
```

---

## 📚 Recursos e Referências

### Papers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings](https://arxiv.org/abs/2402.03216)

### Libraries
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://python.langchain.com/) (framework RAG alternativo)
- [LlamaIndex](https://www.llamaindex.ai/) (framework RAG alternativo)

### Blogs
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [OpenAI Embeddings Best Practices](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)

---

## 🎓 Conceitos-Chave para Estudar

1. **Vector Embeddings**: Como texto é convertido em vetores
2. **Cosine Similarity**: Métrica de similaridade vetorial
3. **FAISS**: Busca eficiente em alta dimensionalidade
4. **Prompt Engineering**: Construção de prompts efetivos
5. **Context Window**: Limites de tokens em LLMs
6. **Temperature**: Controle de aleatoriedade em gerações
7. **Retrieval Metrics**: Recall, Precision, MRR, NDCG
8. **Hallucination**: Quando LLM gera informação não baseada no contexto

---

**Autor**: AI Engineering Team  
**Última Atualização**: 2025  
**Versão**: 1.0