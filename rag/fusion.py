import math
import logging
from collections import defaultdict
from typing import List, Dict, Tuple

class RAGFusion:
    def __init__(self, documents: List[str], word_index: Dict[str, List[int]]):
        self.documents = documents
        self.word_index = word_index

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        logging.info(f"Performing vector search for query: '{query}'")
        query_words = self._tokenize(query)
        doc_scores = defaultdict(float)
        for word in query_words:
            if word in self.word_index:
                idf = math.log(len(self.documents) / len(self.word_index[word]))
                for doc_id in self.word_index[word]:
                    doc_words = self._tokenize(self.documents[doc_id])
                    tf = doc_words.count(word) / len(doc_words)
                    doc_scores[doc_id] += tf * idf
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"Top {top_k} results for '{query}': {sorted_docs[:top_k]}")
        return sorted_docs[:top_k]

    def reciprocal_rank_fusion(self, query_results: Dict[str, List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
        logging.info("Applying Reciprocal Rank Fusion to combine multi-query results.")
        doc_scores = defaultdict(float)
        for query, results in query_results.items():
            for rank, (doc_id, _) in enumerate(results):
                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc_id] += rrf_score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"Fused results: {sorted_docs}")
        return sorted_docs

    def search_and_fuse(self, queries: List[str], top_k: int = 5, rrf_k: int = 60) -> Dict:
        query_results = {}
        for query in queries:
            results = self.vector_search(query, top_k)
            if results:
                query_results[query] = results
        fused_results = self.reciprocal_rank_fusion(query_results, k=rrf_k)
        final_results = []
        for doc_id, rrf_score in fused_results[:top_k]:
            final_results.append({
                'document_id': doc_id,
                'content': self.documents[doc_id],
                'rrf_score': round(rrf_score, 4)
            })
        logging.info(f"Final fused results: {final_results}")
        return {
            'generated_queries': queries,
            'individual_results': {q: [(doc_id, round(score, 4)) for doc_id, score in results] for q, results in query_results.items()},
            'fused_results': final_results
        }

