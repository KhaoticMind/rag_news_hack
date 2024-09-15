import feedparser

from agentutil import instantiate_from_config
from agentutil.chunk import SpacyChunk
from agentutil.datastore import JSONStore
from agentutil.indexer import BaseIndexer
from agentutil.loader import TrafilaturaHTTPLoader
from agentutil.ragstore import PostgresPgVectorRAGDatabase


def get_rss_urls(rss_feed_url):
    # Parse the RSS feed from the provided URL
    feed = feedparser.parse(rss_feed_url)

    # Extract all URLs from the feed
    urls = [entry.link for entry in feed.entries if 'link' in entry]

    return urls

def index(store: str = 'postgres'):

    config_store = JSONStore('json_config_store')

    rag_store:PostgresPgVectorRAGDatabase = instantiate_from_config(config_store.get_config('ragstore', store), config_store)

    http_loader = TrafilaturaHTTPLoader()
    chuncker = SpacyChunk()

    indexer_pg = BaseIndexer(
        loader=http_loader,
        rag_store=rag_store,
        chuncker=chuncker)

    rss_urls = [
        #'http://rss.uol.com.br/feed/economia.xml',
        #'https://rss.uol.com.br/feed/noticias.xml',
        #'https://g1.globo.com/rss/g1/',
        #'https://g1.globo.com/rss/g1/tecnologia',
        #'https://g1.globo.com/rss/g1/politica/'
        #'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362',
        'https://feeds.nbcnews.com/nbcnews/public/news',    
        #'https://abcnews.go.com/abcnews/internationalheadlines',
    ]

    # rag_store_pg.reset_store()
    urls = []
    for rss in rss_urls:
        urls.extend(get_rss_urls(rss))
    # urls = []
    for u in urls:        
        res = rag_store.get({'url': u})
        if len(res) > 0:
            yield f'{u}\n\tALREADY INDEXED'
        else:
            yield f'{u}\n\tINDEXING'
            indexer_pg.index(u)


    rag_store.close()
