"""
Simple Multi-Query RAG Fusion Implementation
A simplified version of RAG fusion without complex dependencies
"""

import re
import math
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import json
import logging
import colorlog

class SimpleRAGFusion:
    def __init__(self):
        # Set up colorful logging
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(asctime)s [%(levelname)s] %(message)s'))
        logging.root.handlers = [handler]
        logging.root.setLevel(logging.INFO)
        logging.info("Initializing SimpleRAGFusion with travel guide documents.")
        # Sample document collection (Travel Guide Use Case)
        self.documents = [
            "Paris is known for its iconic Eiffel Tower, world-class museums, and exquisite cuisine.",
            "Tokyo offers a blend of modern skyscrapers, historic temples, and cherry blossom festivals.",
            "New York City is famous for Times Square, Central Park, and its vibrant nightlife.",
            "Sydney features the stunning Opera House, beautiful beaches, and a lively harbor.",
            "Rome boasts ancient ruins like the Colosseum and delicious Italian food.",
            "Cape Town is renowned for Table Mountain, scenic coastlines, and rich cultural heritage.",
            "Bangkok is a bustling city with ornate shrines, street markets, and vibrant street food.",
            "Barcelona is celebrated for its unique architecture, Mediterranean beaches, and tapas bars.",
            "Cairo is home to the Pyramids of Giza, ancient history, and the Nile River.",
            "Rio de Janeiro is famous for its Carnival festival, Christ the Redeemer statue, and beaches."
        ]
        
        # Simple word frequency index
        self.word_index = self._build_word_index()
    
    def _build_word_index(self) -> Dict[str, List[int]]:
        logging.info("Building word index for documents.")
        index = defaultdict(list)
        
        for i, doc in enumerate(self.documents):
            words = self._tokenize(doc.lower())
            for word in set(words):  # Use set to avoid duplicate entries
                index[word].append(i)
        
        return dict(index)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def generate_similar_queries(self, original_query: str) -> List[str]:
        logging.info(f"Generating similar queries for: '{original_query}'")
        queries = [original_query]
        
        # Add more diverse query variations for travel use case
        words = self._tokenize(original_query)
        
        if len(words) > 1:
            # Reorder words
            queries.append(' '.join(reversed(words)))
            
            # Add city/country/attraction variations
            city_synonyms = {
                'paris': ['france', 'eiffel tower'],
                'tokyo': ['japan', 'shibuya'],
                'new york': ['nyc', 'manhattan'],
                'sydney': ['australia', 'opera house'],
                'rome': ['italy', 'colosseum'],
                'cape town': ['south africa', 'table mountain'],
                'bangkok': ['thailand', 'grand palace'],
                'barcelona': ['spain', 'sagrada familia'],
                'cairo': ['egypt', 'pyramids'],
                'rio': ['brazil', 'copacabana']
            }
            
            for city, syns in city_synonyms.items():
                if city in original_query.lower():
                    for syn in syns:
                        queries.append(original_query + f' {syn}')

        # Add question variations
        if not original_query.lower().startswith(('what', 'how', 'why', 'when', 'where')):
            queries.append(f"what to see in {original_query}")
            queries.append(f"top attractions in {original_query}")
            queries.append(f"travel tips for {original_query}")

        logging.info(f"Generated {len(set(queries))} queries: {set(queries)}")
        return list(set(queries))

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        logging.info(f"Performing vector search for query: '{query}'")
        query_words = self._tokenize(query.lower())
        doc_scores = defaultdict(float)
        
        for word in query_words:
            if word in self.word_index:
                # Simple TF-IDF approximation
                idf = math.log(len(self.documents) / len(self.word_index[word]))
                
                for doc_id in self.word_index[word]:
                    doc_words = self._tokenize(self.documents[doc_id].lower())
                    tf = doc_words.count(word) / len(doc_words)
                    doc_scores[doc_id] += tf * idf
        
        # Sort by score and return top_k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"Top {top_k} results for '{query}': {sorted_docs[:top_k]}")
        return sorted_docs[:top_k]
    
    def reciprocal_rank_fusion(self, query_results: Dict[str, List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
        logging.info("Applying Reciprocal Rank Fusion to combine multi-query results.")
        doc_scores = defaultdict(float)
        
        for query, results in query_results.items():
            for rank, (doc_id, score) in enumerate(results):
                # RRF formula: 1 / (k + rank + 1)
                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc_id] += rrf_score
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"Fused results: {sorted_docs}")
        return sorted_docs
    
    def search_and_fuse(self, original_query: str, top_k: int = 5) -> Dict:
        logging.info(f"Starting RAG Fusion pipeline for query: '{original_query}'")
        # Step 1: Generate multiple queries
        queries = self.generate_similar_queries(original_query)
        
        # Step 2: Perform vector search for each query
        query_results = {}
        for query in queries:
            results = self.vector_search(query, top_k)
            if results:  # Only include queries that returned results
                query_results[query] = results
        
        # Step 3: Apply Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(query_results, k=60)
        
        # Step 4: Prepare final results
        final_results = []
        for doc_id, rrf_score in fused_results[:top_k]:
            final_results.append({
                'document_id': doc_id,
                'content': self.documents[doc_id],
                'rrf_score': round(rrf_score, 4)
            })
        
        logging.info(f"Final fused results: {final_results}")
        return {
            'original_query': original_query,
            'generated_queries': queries,
            'individual_results': {q: [(doc_id, round(score, 4)) for doc_id, score in results] 
                                 for q, results in query_results.items()},
            'fused_results': final_results
        }
    
    def print_results(self, results: Dict):
        print(f"\n{'='*40}\n[RESULTS FOR QUERY] {results['original_query']}\n{'='*40}")
        print(f"Generated Queries ({len(results['generated_queries'])}):")
        for i, query in enumerate(results['generated_queries'], 1):
            print(f"  {i}. {query}")
        
        print(f"\nIndividual Search Results:")
        for query, docs in results['individual_results'].items():
            print(f"\n  Query: '{query}'")
            for doc_id, score in docs:
                print(f"    Doc {doc_id} (score: {score}): {self.documents[doc_id][:60]}...")
        
        print(f"\nFused Results (Top {len(results['fused_results'])}):")
        for i, result in enumerate(results['fused_results'], 1):
            print(f"  {i}. [RRF: {result['rrf_score']}] {result['content']}")


def main():
    """Example usage"""
    rag_fusion = SimpleRAGFusion()
    
    # More complex, real-world queries for travel use case
    test_queries = [
        "paris travel",
        "things to do in tokyo",
        "best beaches in sydney",
        "rome attractions",
        "cape town sightseeing"
    ]
    
    for query in test_queries:
        results = rag_fusion.search_and_fuse(query)
        rag_fusion.print_results(results)
        print()


if __name__ == "__main__":
    main()
