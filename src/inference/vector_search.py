"""
Vector Search and Retrieval using FAISS
Implements efficient similarity search for recommendations
"""

import faiss
import numpy as np
import pickle
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import time
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result with item metadata"""
    item_id: str
    score: float
    metadata: Dict


class FAISSVectorSearch:
    """
    FAISS-based vector search engine for fast similarity retrieval
    """
    
    def __init__(
        self,
        dimension: int = 128,
        index_type: str = "IVF1024,Flat",
        metric: str = "L2",
        nprobe: int = 32
    ):
        """
        Initialize FAISS index
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type (e.g., "IVF1024,Flat", "HNSW32")
            metric: Distance metric ("L2" or "IP" for inner product)
            nprobe: Number of clusters to search (for IVF indices)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.nprobe = nprobe
        
        self.index = None
        self.item_ids = []
        self.metadata = {}
        
        self.logger = logging.getLogger(__name__)
        
    def build_index(
        self,
        embeddings: np.ndarray,
        item_ids: List[str],
        metadata: Optional[Dict[str, Dict]] = None
    ):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: Array of shape (N, dimension)
            item_ids: List of item identifiers
            metadata: Optional metadata for each item
        """
        self.logger.info(f"Building FAISS index with {len(embeddings)} vectors")
        
        # Validate inputs
        assert embeddings.shape[0] == len(item_ids), "Mismatch between embeddings and item_ids"
        assert embeddings.shape[1] == self.dimension, f"Expected dimension {self.dimension}, got {embeddings.shape[1]}"
        
        # Normalize embeddings if using inner product
        if self.metric == "IP":
            faiss.normalize_L2(embeddings)
        
        # Create index
        if self.metric == "L2":
            quantizer = faiss.IndexFlatL2(self.dimension)
        else:
            quantizer = faiss.IndexFlatIP(self.dimension)
        
        # Parse index type and create appropriate index
        if self.index_type.startswith("IVF"):
            # Extract number of clusters
            n_clusters = int(self.index_type.split("IVF")[1].split(",")[0])
            
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
            
            # Train the index
            self.logger.info(f"Training IVF index with {n_clusters} clusters")
            self.index.train(embeddings)
            
            # Set nprobe
            self.index.nprobe = self.nprobe
            
        elif self.index_type.startswith("HNSW"):
            # HNSW index for high-dimensional data
            M = int(self.index_type.split("HNSW")[1])
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            
        else:
            # Flat index (exact search)
            self.index = quantizer
        
        # Add vectors to index
        self.logger.info("Adding vectors to index")
        self.index.add(embeddings)
        
        # Store metadata
        self.item_ids = item_ids
        self.metadata = metadata or {}
        
        self.logger.info(f"Index built successfully with {self.index.ntotal} vectors")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 20,
        filter_fn: Optional[callable] = None
    ) -> List[SearchResult]:
        """
        Search for similar items
        
        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            top_k: Number of results to return
            filter_fn: Optional filtering function
        
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Reshape query if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize if using inner product
        if self.metric == "IP":
            faiss.normalize_L2(query_embedding)
        
        # Perform search
        start_time = time.time()
        
        # Search for more results if filtering
        search_k = top_k * 5 if filter_fn else top_k
        
        distances, indices = self.index.search(query_embedding, search_k)
        
        search_time = (time.time() - start_time) * 1000
        
        # Convert to SearchResult objects
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            item_id = self.item_ids[idx]
            
            # Convert distance to similarity score
            if self.metric == "L2":
                score = 1.0 / (1.0 + dist)
            else:  # IP
                score = float(dist)
            
            # Get metadata
            item_metadata = self.metadata.get(item_id, {})
            
            # Apply filter if provided
            if filter_fn and not filter_fn(item_id, item_metadata):
                continue
            
            results.append(SearchResult(
                item_id=item_id,
                score=score,
                metadata=item_metadata
            ))
            
            if len(results) >= top_k:
                break
        
        self.logger.debug(f"Search completed in {search_time:.2f}ms, returned {len(results)} results")
        
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 20
    ) -> List[List[SearchResult]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Array of shape (N, dimension)
            top_k: Number of results per query
        
        Returns:
            List of lists of SearchResult objects
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize if using inner product
        if self.metric == "IP":
            faiss.normalize_L2(query_embeddings)
        
        # Perform batch search
        start_time = time.time()
        distances, indices = self.index.search(query_embeddings, top_k)
        search_time = (time.time() - start_time) * 1000
        
        # Convert to SearchResult objects
        all_results = []
        for query_dists, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_dists, query_indices):
                if idx == -1:
                    continue
                
                item_id = self.item_ids[idx]
                
                if self.metric == "L2":
                    score = 1.0 / (1.0 + dist)
                else:
                    score = float(dist)
                
                results.append(SearchResult(
                    item_id=item_id,
                    score=score,
                    metadata=self.metadata.get(item_id, {})
                ))
            
            all_results.append(results)
        
        self.logger.debug(
            f"Batch search completed in {search_time:.2f}ms, "
            f"processed {len(query_embeddings)} queries"
        )
        
        return all_results
    
    def add_items(
        self,
        embeddings: np.ndarray,
        item_ids: List[str],
        metadata: Optional[Dict[str, Dict]] = None
    ):
        """
        Add new items to existing index
        
        Args:
            embeddings: Array of shape (N, dimension)
            item_ids: List of item identifiers
            metadata: Optional metadata for each item
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize if using inner product
        if self.metric == "IP":
            faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Update metadata
        self.item_ids.extend(item_ids)
        if metadata:
            self.metadata.update(metadata)
        
        self.logger.info(f"Added {len(item_ids)} items. Total: {self.index.ntotal}")
    
    def save(self, index_path: str, metadata_path: str):
        """
        Save index and metadata to disk
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        self.logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_dict = {
            'item_ids': self.item_ids,
            'metadata': self.metadata,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metric': self.metric,
            'nprobe': self.nprobe
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f)
        
        self.logger.info(f"Saved metadata to {metadata_path}")
    
    def load(self, index_path: str, metadata_path: str):
        """
        Load index and metadata from disk
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        self.logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata_dict = pickle.load(f)
        
        self.item_ids = metadata_dict['item_ids']
        self.metadata = metadata_dict['metadata']
        self.dimension = metadata_dict['dimension']
        self.index_type = metadata_dict['index_type']
        self.metric = metadata_dict['metric']
        self.nprobe = metadata_dict.get('nprobe', 32)
        
        # Set nprobe if IVF index
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        self.logger.info(f"Loaded metadata from {metadata_path}")
        self.logger.info(f"Index contains {self.index.ntotal} vectors")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            return {"status": "Index not built"}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    dimension = 128
    n_items = 10000
    embeddings = np.random.randn(n_items, dimension).astype('float32')
    item_ids = [f"item_{i}" for i in range(n_items)]
    
    # Create metadata
    categories = ['Fashion', 'Tech', 'Home', 'Food', 'Travel']
    metadata = {
        item_id: {'category': np.random.choice(categories)}
        for item_id in item_ids
    }
    
    # Build index
    search_engine = FAISSVectorSearch(dimension=dimension)
    search_engine.build_index(embeddings, item_ids, metadata)
    
    # Search
    query = np.random.randn(dimension).astype('float32')
    results = search_engine.search(query, top_k=10)
    
    print(f"\nTop 10 results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.item_id} - Score: {result.score:.4f} - {result.metadata}")
    
    # Save index
    search_engine.save("test_index.faiss", "test_metadata.pkl")
    
    print(f"\nIndex stats: {search_engine.get_stats()}")
