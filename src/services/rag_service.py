from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from datetime import datetime
import logging
from src.config.settings import settings
from src.models.core import RagDocument

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.client = chromadb.Client(
            ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=settings.chroma_persist_directory
            )
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={
                "hnsw:space": settings.rag.distance_metric,
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 100
            }
        )
        
    async def add_document(self, document: RagDocument) -> bool:
        try:
            self.collection.add(
                documents=[document.content],
                metadatas=[document.metadata],
                ids=[document.id],
                embeddings=[document.embedding] if document.embedding else None
            )
            return True
        except Exception as e:
            logger.error(f"RAG add document error: {e}")
            return False
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k or settings.rag.top_k,
                where=filter_metadata,
                include_metadata=True
            )
            
            # Enhanced ranking with additional metrics
            ranked_results = self._rank_results(
                results,
                query,
                settings.rag.similarity_threshold
            )
            
            return ranked_results
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return []
    
    def _rank_results(
        self,
        results: Dict,
        query: str,
        threshold: float
    ) -> List[Dict]:
        ranked_docs = []
        
        for idx, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            if distance > threshold:
                continue
                
            # Calculate additional relevance metrics
            relevance_score = self._calculate_relevance(
                query,
                doc,
                distance,
                metadata
            )
            
            ranked_docs.append({
                'content': doc,
                'metadata': metadata,
                'relevance_score': relevance_score,
                'original_rank': idx
            })
        
        # Sort by relevance score
        ranked_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        return ranked_docs
    
    def _calculate_relevance(
        self,
        query: str,
        document: str,
        distance: float,
        metadata: Dict
    ) -> float:
        # Implement sophisticated relevance calculation
        # This is a simplified example
        base_score = 1 - distance  # Convert distance to similarity
        
        # Consider document freshness if timestamp exists
        freshness_score = 0.0
        if 'timestamp' in metadata:
            age_days = (datetime.now() - datetime.fromisoformat(metadata['timestamp'])).days
            freshness_score = max(0, 1 - (age_days / 365))  # Decay over a year
        
        # Consider exact match bonus
        exact_match_bonus = 0.2 if query.lower() in document.lower() else 0.0
        
        # Consider metadata quality
        metadata_score = min(0.1 * len(metadata), 0.3)  # Up to 0.3 bonus for rich metadata
        
        # Weighted combination
        final_score = (
            0.6 * base_score +
            0.2 * freshness_score +
            0.1 * exact_match_bonus +
            0.1 * metadata_score
        )
        
        return final_score
    
    async def delete_document(self, document_id: str) -> bool:
        try:
            self.collection.delete(ids=[document_id])
            return True
        except Exception as e:
            logger.error(f"RAG delete document error: {e}")
            return False
    
    async def update_document(self, document: RagDocument) -> bool:
        try:
            # ChromaDB doesn't have direct update, so we delete and re-add
            await self.delete_document(document.id)
            return await self.add_document(document)
        except Exception as e:
            logger.error(f"RAG update document error: {e}")
            return False