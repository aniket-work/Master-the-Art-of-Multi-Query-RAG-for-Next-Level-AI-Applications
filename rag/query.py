import logging
from typing import List, Dict

class QueryGenerator:
    def __init__(self, city_synonyms: Dict[str, List[str]]):
        self.city_synonyms = city_synonyms

    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def generate(self, original_query: str) -> List[str]:
        logging.info(f"Generating similar queries for: '{original_query}'")
        queries = [original_query]
        words = self._tokenize(original_query)
        if len(words) > 1:
            queries.append(' '.join(reversed(words)))
            for city, syns in self.city_synonyms.items():
                if city in original_query.lower():
                    for syn in syns:
                        queries.append(original_query + f' {syn}')
        if not original_query.lower().startswith(('what', 'how', 'why', 'when', 'where')):
            queries.append(f"what to see in {original_query}")
            queries.append(f"top attractions in {original_query}")
            queries.append(f"travel tips for {original_query}")
        logging.info(f"Generated {len(set(queries))} queries: {set(queries)}")
        return list(set(queries))

