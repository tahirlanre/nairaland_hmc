import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import argparse
import logging

from get_topic_posts import parse_post
from utils import init_logger

init_logger()

# Set headers
heads = requests.utils.default_headers()
heads.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forum", required=True, help="Forum to scrape")
    args = parser.parse_args()

    return args

def get_topic_id(topic):
    _id = ''
    name = topic.find('a').get('name')
    if name not in ('top', None):
        _id = name
    return _id

def get_topic_title(topic):
    title = ''
    b_tag = topic.find('b')
    if b_tag:
        title = b_tag.get_text()
    return title

def get_topic_length(topic):
    length = ''
    span_tag = topic.find('span', class_='s')
    if span_tag:
        length_tag = span_tag.find_all('b')[1]
        if length_tag:
            length = length_tag.get_text()
    return length

def get_topic_view(topic):
    views = ''
    span_tag = topic.find('span', class_='s')
    if span_tag:
        view_tag = span_tag.find_all('b')[2]
        if view_tag:
            views = view_tag.get_text()
    return views

def get_topic_author(topic):
    author = ''
    span_tag = topic.find('span', class_='s')
    if span_tag:
        author_tag = span_tag.find('a')
        if author_tag:
            author = author_tag.get_text()
    return author

def get_topic_info(t):
    topic = {}
    topic['topic_id'] = get_topic_id(t)
    topic['title'] = get_topic_title(t)
    topic['length'] = get_topic_length(t)
    topic['views'] = get_topic_view(t)
    topic['author'] = get_topic_author(t)
    return topic

def parse_forum(args):
    forum = args.forum

    logging.info(f"******    Getting topics from {forum} forum   ******")

    start_url = 'https://www.nairaland.com/{}/posts/'.format(forum)
    r1 = requests.get(start_url, heads)
    raw_html = BeautifulSoup(r1.text, 'html5lib')


    # retrieve the number of pages in the forum
    page = int(raw_html.select('body > div > p:nth-child(7)')[0].select('b')[1].text) 

    counter = 0
    topic_tags = []
    with open(f"data/{forum}_topics.json", "w") as f:
        for i in range(page):
            next_page = start_url + '{}'.format(i)
            r2 = requests.get(next_page, heads)
            forum_html = BeautifulSoup(r2.text, 'html5lib')
            topic_tags = forum_html.find_all('td', attrs = {'id': True})
            for tag in topic_tags:
                topic = get_topic_info(tag)
                f.write(json.dumps(topic)+"\n")
                counter += 1
            logging.info(f"No of pages: {i+1},   No of topics: {counter}")

def main():
    """ Runs web scraping scripts to collect data from the site and
        copy into (../raw) 
    """
    args = parse_args()
    parse_forum(args)    
    
if __name__ == '__main__':
    main()