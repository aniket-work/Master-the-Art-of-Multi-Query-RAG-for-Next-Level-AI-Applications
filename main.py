import logging
from rag.data_loader import load_documents, load_settings
from rag.logging_config import setup_logging
from rag.index import DocumentIndexer
from rag.query import QueryGenerator
from rag.fusion import RAGFusion


def print_results(results, documents, original_query):
    print(f"\n{'='*40}\n[RESULTS FOR QUERY] {original_query}\n{'='*40}")
    print(f"Generated Queries ({len(results['generated_queries'])}):")
    for i, query in enumerate(results['generated_queries'], 1):
        print(f"  {i}. {query}")
    print(f"\nIndividual Search Results:")
    for query, docs in results['individual_results'].items():
        print(f"\n  Query: '{query}'")
        for doc_id, score in docs:
            print(f"    Doc {doc_id} (score: {score}): {documents[doc_id][:60]}...")
    print(f"\nFused Results (Top {len(results['fused_results'])}):")
    for i, result in enumerate(results['fused_results'], 1):
        print(f"  {i}. [RRF: {result['rrf_score']}] {result['content']}")

def main():
    settings = load_settings('config/settings.yaml')
    setup_logging(settings)
    documents = load_documents('config/documents.json')
    indexer = DocumentIndexer(documents)
    query_gen = QueryGenerator(settings['city_synonyms'])
    fusion = RAGFusion(documents, indexer.word_index)
    test_queries = [
        "paris travel",
        "things to do in tokyo",
        "best beaches in sydney",
        "rome attractions",
        "cape town sightseeing"
    ]
    for original_query in test_queries:
        queries = query_gen.generate(original_query)
        results = fusion.search_and_fuse(
            queries,
            top_k=settings['search']['top_k'],
            rrf_k=settings['search']['rrf_k']
        )
        print_results(results, documents, original_query)
        print()

if __name__ == "__main__":
    main()

