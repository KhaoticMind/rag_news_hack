from datetime import datetime

import requests
#from bs4 import BeautifulSoup
from trafilatura import extract, extract_metadata, fetch_url

from .base import DocLoader, LoadedData


class SimpleHTTPLoader(DocLoader):
    def load(self, source, **kwargs) -> list[LoadedData]:
        response = requests.get(source)
        html_content = response.content

        # Parse the cleaned main content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.title.text
        soup = soup.body

        # Optionally, remove unwanted sections like footer, nav, ads
        for tag in soup(['nav', 'footer', 'aside', 'script', 'style', 'meta']):
            tag.decompose()

        content = soup.get_text(separator=' ', strip=True)

        data = LoadedData(content=content, metadata={
                          'url': source, 'retrieval_data': datetime.now().isoformat(), 'title': title})
        return [data]


class TrafilaturaHTTPLoader(DocLoader):
    def load(self, source, **kwargs) -> list[LoadedData]:

        downloaded = fetch_url(source)
        content = extract(downloaded)
        title = extract_metadata(downloaded).title

        if content:
            data = LoadedData(content=content, metadata={
                          'url': source, 'retrieval_data': datetime.now().isoformat(), 'title': title})
            return [data]
        else:
            return []
