import feedparser
from dotenv import load_dotenv

from agentutil import instantiate_from_config
from agentutil.chunk.spacy import SpacyChunk
from agentutil.datastore.json import JSONStore
from agentutil.indexer.base import BaseIndexer
from agentutil.loader.http import TrafilaturaHTTPLoader
from agentutil.ragstore.postgres import PostgresPgVectorRAGDatabase

load_dotenv()


def get_rss_urls(rss_feed_url):
    # Parse the RSS feed from the provided URL
    feed = feedparser.parse(rss_feed_url)

    # Extract all URLs from the feed
    urls = [entry.link for entry in feed.entries if 'link' in entry]

    return urls


config_store = JSONStore('json_config_store')

pg_store_config = config_store.get_config('ragstore', 'postgres')
rag_store_pg = instantiate_from_config(pg_store_config, config_store)

http_loader = TrafilaturaHTTPLoader()
chuncker = SpacyChunk()

indexer_pg = BaseIndexer(
    loader=http_loader,
    rag_store=rag_store_pg,
    chuncker=chuncker)

rss_urls = [
    'http://rss.uol.com.br/feed/economia.xml',
    'https://rss.uol.com.br/feed/noticias.xml',
    'https://g1.globo.com/rss/g1/',
    'https://g1.globo.com/rss/g1/tecnologia',
    'https://g1.globo.com/rss/g1/politica/'
    'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362',
    'https://feeds.nbcnews.com/nbcnews/public/news',    
    'https://abcnews.go.com/abcnews/internationalheadlines',
]

# rag_store_pg.reset_store()
urls = []
for rss in rss_urls:
    urls.extend(get_rss_urls(rss))
# urls = []
for u in urls:
    print(u)
    res = rag_store_pg.get({'url': u})
    if len(res) > 0:
        print(f'\tALREADY INDEXED')
    else:
        print(f'\tINDEXING')
        indexer_pg.index(u)


rag_store_pg.close()
