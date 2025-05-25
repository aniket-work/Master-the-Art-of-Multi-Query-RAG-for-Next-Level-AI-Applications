import re
from collections import defaultdict
from typing import List, Dict

class DocumentIndexer:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.word_index = self._build_word_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def _build_word_index(self) -> Dict[str, List[int]]:
        index = defaultdict(list)
        for i, doc in enumerate(self.documents):
            words = self._tokenize(doc)
            for word in set(words):
                index[word].append(i)
        return dict(index)

