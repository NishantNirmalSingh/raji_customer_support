## 2025-05-15 - [Consolidated Embeddings & Caching]
**Learning:** Redundant initializations of the same Transformer model (e.g., `HuggingFaceEmbeddings`) are common in RAG-based systems and can double memory usage and slow down startup time. Implementing a simple `mtime`-based cache for CSV-backed data is a low-risk, high-impact optimization for Admin Dashboards.
**Action:** Always check for duplicate model initializations across RAG and ticketing systems. Use shared global instances for embedding models. Apply `os.path.getmtime` checks for file-backed data structures to minimize Disk I/O.
