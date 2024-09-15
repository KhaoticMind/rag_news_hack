from typing import List

import spacy

from .base import BaseChunk

# Function to get the number of tokens using tiktoken. Keep in here for future use
# def num_tokens_from_string(string: str) -> int:
#    encoding = tiktoken.encoding_for_model("gpt-4o")
#    num_tokens = len(encoding.encode(string))
#    return num_tokens


class SpacyChunk(BaseChunk):
    @staticmethod
    def split(text: str, chunk_chars_length: int = 1500) -> list[str]:
        # Load spaCy language model
        try:
            nlp = spacy.load("pt_core_news_sm")
        except:
            import pip
            pip.main(['install', 'https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0-py3-none-any.whl'])
            nlp = spacy.load("pt_core_news_sm")

        doc = nlp(text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_chunk_chars: int = 0

        for sent in doc.sents:
            sent_striped = sent.text.strip()
            #lets ignore empty sentences
            if sent_striped == '':
                continue
            # sent_tokens = num_tokens_from_string(sent.text)#len(nlp(sent.text))  # Count tokens in sentence
            sent_chars = len(sent_striped)#len(nlp(sent_striped))
            if current_chunk_chars + sent_chars > chunk_chars_length:
                # Join sentences with space
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_chunk_chars = 0

            current_chunk.append(sent_striped)
            current_chunk_chars += sent_chars

        # Add any remaining sentences
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
