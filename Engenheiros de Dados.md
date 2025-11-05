# Documentação para Engenheiros de Dados - Sistema RAG

## 🎯 Objetivo

Este documento descreve a arquitetura de dados, pipeline de ingestão, modelagem e operações do sistema RAG no Databricks.

---

## 📐 Arquitetura de Dados

### Unity Catalog Structure

```
bronze (catalog)
└── default (schema)
    ├── docs_raw          # Documentos brutos
    ├── docs_chunks       # Chunks processados
    ├── docs_embeddings   # Vetores embedding
    └── rag_audit         # Logs de auditoria
```

### Volumes

```
/Volumes/bronze/default/documentos_agent/
└── [Documentos fonte: PDF, DOCX, PPTX]
```

---

## 📊 Modelagem de Dados

### 1. Tabela `docs_raw`

**Propósito**: Armazenar documentos brutos processados

```sql
CREATE TABLE bronze.default.docs_raw (
    doc_id STRING,           -- UUID único do documento
    file_name STRING,        -- Nome do arquivo original
    file_type STRING,        -- Extensão: pdf, docx, pptx
    text_content STRING,     -- Texto extraído completo
    char_count BIGINT,       -- Contagem de caracteres
    processed_at TIMESTAMP   -- Data/hora do processamento
)
USING DELTA
PARTITIONED BY (file_type)
```

**Características:**
- Particionada por `file_type` para otimizar queries por tipo
- `doc_id` é gerado via UUID
- `text_content` pode ser grande (use Z-ordering se necessário)

**Exemplo de Insert:**
```python
df_raw = spark.createDataFrame([{
    "doc_id": str(uuid.uuid4()),
    "file_name": "manual.pdf",
    "file_type": "pdf",
    "text_content": "...",
    "char_count": len(text),
    "processed_at": datetime.now(timezone.utc)
}])

df_raw.write.mode("overwrite").saveAsTable(TABLE_DOCS_RAW)
```

---

### 2. Tabela `docs_chunks`

**Propósito**: Armazenar chunks processados com overlap

```sql
CREATE TABLE bronze.default.docs_chunks (
    chunk_id STRING,         -- UUID único do chunk
    doc_id STRING,           -- FK para docs_raw
    chunk_index INT,         -- Índice sequencial do chunk (0, 1, 2...)
    chunk_text STRING,       -- Texto do chunk
    char_count INT           -- Tamanho do chunk
)
USING DELTA
PARTITIONED BY (doc_id)
```

**Características:**
- Particionada por `doc_id` para co-location de chunks do mesmo documento
- `chunk_index` mantém ordem original
- Chunks são gerados com overlap configurável (padrão: 120 caracteres)

**Padrão de Chunking:**
```python
CHUNK_SIZE = 800      # Caracteres por chunk
CHUNK_OVERLAP = 120   # Overlap entre chunks consecutivos
```

**Exemplo de Chunk:**
```
Chunk 0: [0:800]
Chunk 1: [680:1480]   # Overlap de 120 chars
Chunk 2: [1360:2160]
```

---

### 3. Tabela `docs_embeddings`

**Propósito**: Armazenar vetores de embedding de cada chunk

```sql
CREATE TABLE bronze.default.docs_embeddings (
    chunk_id STRING,                    -- FK para docs_chunks
    doc_id STRING,                      -- FK para docs_raw
    embedding ARRAY<FLOAT>              -- Vetor de embedding (1024 dims)
)
USING DELTA
PARTITIONED BY (doc_id)
```

**Características:**
- Vetores de 1024 dimensões (modelo BAAI/bge-m3)
- Embeddings são gerados em batch para eficiência
- Particionada por `doc_id` para queries eficientes

**Tamanho Estimado:**
```
Embedding: 1024 floats × 4 bytes = ~4 KB por chunk
1000 chunks = ~4 MB de embeddings
```

**Exemplo de Insert:**
```python
df_embeds = spark.createDataFrame([{
    "chunk_id": chunk_id,
    "doc_id": doc_id,
    "embedding": embedding.tolist()
}])

df_embeds.write.mode("overwrite").saveAsTable(TABLE_EMBEDS)
```

---

### 4. Tabela `rag_audit`

**Propósito**: Auditoria e telemetria de consultas RAG

```sql
CREATE TABLE bronze.default.rag_audit (
    query_id STRING,                    -- UUID da consulta
    timestamp TIMESTAMP,                -- Data/hora da query
    provider STRING,                    -- LLM provider usado
    model STRING,                       -- Modelo específico
    question STRING,                    -- Pergunta do usuário
    top_k INT,                          -- Número de chunks recuperados
    latency_ms DOUBLE,                  -- Latência total em ms
    chunks_used STRING,                 -- JSON dos chunks usados
    answer STRING                       -- Resposta gerada
)
USING DELTA
PARTITIONED BY (DATE(timestamp))
```

**Características:**
- Particionada por data para queries temporais eficientes
- `chunks_used` armazena JSON para análise de relevância
- Permite análise de performance por provider/modelo

**Exemplo de Query Analytics:**
```sql
-- Latência média por provider
SELECT 
    provider,
    AVG(latency_ms) as avg_latency,
    COUNT(*) as total_queries
FROM bronze.default.rag_audit
WHERE DATE(timestamp) >= CURRENT_DATE - 7
GROUP BY provider
ORDER BY avg_latency;

-- Queries mais frequentes
SELECT 
    question,
    COUNT(*) as frequency
FROM bronze.default.rag_audit
GROUP BY question
ORDER BY frequency DESC
LIMIT 10;
```

---

## 🔄 Pipeline de Ingestão

### Fluxo de Dados

```
┌─────────────────┐
│  Upload File    │
│  (Volume)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract Text   │
│  (PyPDF, docx)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Raw Doc   │
│  (docs_raw)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chunk Text     │
│  (800 chars)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Chunks    │
│  (docs_chunks)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generate       │
│  Embeddings     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Save Vectors   │
│  (embeddings)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Build FAISS    │
│  Index          │
└─────────────────┘
```

### Estratégia de Write

**Problema**: Evitar duplicação de dados em re-execuções

**Solução**: Union + Overwrite pattern

```python
# 1. Ler dados existentes (se houver)
try:
    existing_df = spark.read.table(TABLE_NAME)
except:
    existing_df = spark.createDataFrame([], schema)

# 2. Fazer union com novos dados
final_df = existing_df.union(new_df)

# 3. Remover duplicatas (por doc_id ou chunk_id)
final_df = final_df.dropDuplicates(["doc_id"])

# 4. Overwrite completo
final_df.write.mode("overwrite").saveAsTable(TABLE_NAME)
```

**Benefícios:**
- Idempotente (pode re-executar sem efeitos colaterais)
- Evita crescimento infinito de dados
- Mantém histórico necessário

---

## ⚡ Otimizações de Performance

### 1. Z-Ordering

Para queries frequentes em `doc_id`:

```sql
OPTIMIZE bronze.default.docs_chunks 
ZORDER BY (doc_id);

OPTIMIZE bronze.default.docs_embeddings 
ZORDER BY (doc_id);
```

### 2. Vacuum

Limpar versões antigas das tabelas Delta:

```sql
-- Manter apenas últimos 7 dias
VACUUM bronze.default.docs_raw RETAIN 168 HOURS;
VACUUM bronze.default.docs_chunks RETAIN 168 HOURS;
VACUUM bronze.default.docs_embeddings RETAIN 168 HOURS;

-- Auditoria: manter 30 dias
VACUUM bronze.default.rag_audit RETAIN 720 HOURS;
```

### 3. Batch Processing

Embeddings são gerados em batches para eficiência:

```python
BATCH_SIZE = 32  # Processar 32 chunks por vez

for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
    batch = chunks[i:i+BATCH_SIZE]
    embeddings = model.encode(batch, normalize_embeddings=True)
    # Salvar batch
```

### 4. FAISS Index Optimization

```python
# IndexFlatIP (Inner Product) para busca exata
dimension = 1024
index = faiss.IndexFlatIP(dimension)

# Para produção com muitos documentos, considere:
# IndexIVFFlat (aprox. nearest neighbor, mais rápido)
# nlist = 100
# quantizer = faiss.IndexFlatIP(dimension)
# index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
```

---

## 📈 Monitoramento e Qualidade

### Métricas de Pipeline

```python
# Após processamento, coletar métricas:
metrics = {
    "total_docs": df_raw.count(),
    "total_chunks": df_chunks.count(),
    "avg_chunks_per_doc": df_chunks.groupBy("doc_id").count().agg({"count": "avg"}),
    "total_embeddings": df_embeds.count(),
    "processing_time_sec": elapsed_time
}
```

### Data Quality Checks

```python
# 1. Verificar integridade referencial
chunks_without_doc = spark.sql("""
    SELECT chunk_id 
    FROM bronze.default.docs_chunks c
    LEFT JOIN bronze.default.docs_raw d ON c.doc_id = d.doc_id
    WHERE d.doc_id IS NULL
""")

# 2. Verificar embeddings órfãos
embeddings_without_chunk = spark.sql("""
    SELECT e.chunk_id
    FROM bronze.default.docs_embeddings e
    LEFT JOIN bronze.default.docs_chunks c ON e.chunk_id = c.chunk_id
    WHERE c.chunk_id IS NULL
""")

# 3. Verificar chunks vazios ou muito pequenos
invalid_chunks = spark.sql("""
    SELECT chunk_id, char_count
    FROM bronze.default.docs_chunks
    WHERE char_count < 50 OR chunk_text IS NULL
""")
```

### Queries de Análise

```sql
-- Documentos por tipo
SELECT file_type, COUNT(*) as count
FROM bronze.default.docs_raw
GROUP BY file_type;

-- Distribuição de tamanho de chunks
SELECT 
    CASE 
        WHEN char_count < 200 THEN 'Pequeno'
        WHEN char_count < 600 THEN 'Médio'
        ELSE 'Grande'
    END as size_category,
    COUNT(*) as count
FROM bronze.default.docs_chunks
GROUP BY size_category;

-- Análise de auditoria (últimos 7 dias)
SELECT 
    provider,
    DATE(timestamp) as date,
    COUNT(*) as queries,
    AVG(latency_ms) as avg_latency,
    PERCENTILE(latency_ms, 0.95) as p95_latency
FROM bronze.default.rag_audit
WHERE timestamp >= CURRENT_DATE - 7
GROUP BY provider, DATE(timestamp)
ORDER BY date DESC, provider;
```

---

## 🔐 Segurança e Governança

### Unity Catalog Permissions

```sql
-- Conceder acesso de leitura para time de analytics
GRANT SELECT ON TABLE bronze.default.docs_raw TO `analytics_team`;
GRANT SELECT ON TABLE bronze.default.rag_audit TO `analytics_team`;

-- Restringir write access
GRANT SELECT, INSERT, DELETE ON TABLE bronze.default.docs_raw TO `data_engineers`;
```

### Data Lineage

Unity Catalog rastreia automaticamente:
- Origem dos dados (Volumes)
- Transformações aplicadas
- Tabelas downstream
- Acesso e modificações

Visualize em: **Data Explorer > Tabela > Lineage**

---

## 🚀 Escalabilidade

### Para Múltiplos Documentos

```python
# Processar diretório inteiro
import glob

docs_path = "/Volumes/bronze/default/documentos_agent/*.pdf"
files = glob.glob(docs_path)

for file_path in files:
    process_document(file_path)  # Pipeline completo
```

### Para Volume Alto

- Use **Auto Loader** para ingestão contínua
- Implemente **Change Data Capture** em docs_raw
- Configure **Streaming Tables** para processamento em tempo real

```python
# Exemplo com Auto Loader
df = (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .load("/Volumes/bronze/default/documentos_agent/")
)

# Processar stream
df.writeStream \
    .foreachBatch(process_batch) \
    .start()
```

---

## 📝 Manutenção

### Daily Tasks
- Verificar logs de erro no processamento
- Monitorar crescimento das tabelas
- Revisar queries lentas na auditoria

### Weekly Tasks
- Executar OPTIMIZE com Z-ORDER
- Analisar métricas de qualidade de chunks
- Revisar performance dos embeddings

### Monthly Tasks
- Executar VACUUM para limpeza
- Revisar e arquivar dados de auditoria antigos
- Análise de uso por provider/modelo

---

## 🛠️ Troubleshooting

### Problema: Chunks duplicados

```sql
-- Identificar duplicatas
SELECT chunk_id, COUNT(*) as count
FROM bronze.default.docs_chunks
GROUP BY chunk_id
HAVING COUNT(*) > 1;

-- Resolver: usar dropDuplicates no pipeline
```

### Problema: Embeddings faltando

```sql
-- Encontrar chunks sem embedding
SELECT c.chunk_id
FROM bronze.default.docs_chunks c
LEFT JOIN bronze.default.docs_embeddings e ON c.chunk_id = e.chunk_id
WHERE e.chunk_id IS NULL;

-- Reprocessar esses chunks
```

### Problema: Performance degradada

```
1. Executar ANALYZE TABLE para atualizar estatísticas
2. Verificar Z-ordering das tabelas
3. Analisar query plans com EXPLAIN
4. Considerar particionamento adicional
```

---

## 📚 Referências

- [Delta Lake Best Practices](https://docs.delta.io/latest/best-practices.html)
- [Unity Catalog Documentation](https://docs.databricks.com/unity-catalog/)
- [OPTIMIZE and Z-ORDER](https://docs.databricks.com/delta/optimizations.html)
- [VACUUM](https://docs.databricks.com/delta/vacuum.html)

---

**Autor**: Data Engineering Team  
**Última Atualização**: 2025  
**Versão**: 1.0