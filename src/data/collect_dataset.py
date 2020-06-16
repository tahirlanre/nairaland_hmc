import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm
import click
from datetime import datetime

# Set headers
heads = requests.utils.default_headers()
heads.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
})

@click.command()
@click.argument('forums', nargs=-1)
def main(forums):
    """ Runs web scraping scripts to collect data from the site and
        copy into (../raw) 
    """
    for forum in forums:
        res = parse_forum(forum)
        df = pd.DataFrame(res)
        df.to_csv('data/raw/{}.csv'.format(forum))

def parse_forum(forum):
    start_url = 'https://www.nairaland.com/{}/posts/'.format(forum)
    r1 = requests.get(start_url, heads)
    raw_html = BeautifulSoup(r1.text, 'html5lib')

    # create empty df to store posts from parsed topics in the forum
    df_posts = pd.DataFrame()

    topics = []

    # retrieve the number of pages in the forum
    page = int(raw_html.select('body > div > p:nth-child(7)')[0].select('b')[1].text) 
    
    for i in tqdm(range(page)):
        next_page = start_url + '{}'.format(i)
        r2 = requests.get(next_page, heads)
        forum_html = BeautifulSoup(r2.text, 'html5lib')
        topic_tags = forum_html.find_all('td', attrs = {'id': True})
        for tag in topic_tags:
            topic = get_topic(tag)
            topics.append(topic)
    
    df_topics = pd.DataFrame(topics)
    df_topics.to_csv('data/raw/{}_topics.csv'.format(forum))

    iter = 0
    for topic in topics:
        res = parse_topic(topic['topic_id'])
        df_res = pd.DataFrame(res)
        df_res['forum'] = forum
        df_posts = df_posts.append(df_res)

        iter += 1
        # save after every 500 pages
        if iter % 500 == 0:
            df_posts.to_csv('data/raw/{}_{}.csv'.format(forum, iter))
    return df_posts

def get_topic(t):
    topic = {}
    topic['topic_id'] = getTopicId(t)
    topic['title'] = getTopicTitle(t)
    topic['length'] = getTopicLength(t)
    topic['views'] = getTopicView(t)
    topic['author'] = getTopicAuthor(t)
    return topic

def getTopicId(topic):
    _id = ''
    name = topic.find('a').get('name')
    if name not in ('top', None):
        _id = name
    return _id

def getTopicTitle(topic):
    title = ''
    b_tag = topic.find('b')
    if b_tag:
        title = b_tag.get_text()
    return title

def getTopicLength(topic):
    length = ''
    span_tag = topic.find('span', class_='s')
    if span_tag:
        length_tag = span_tag.find_all('b')[1]
        if length_tag:
            length = length_tag.get_text()
    return length

def getTopicView(topic):
    views = ''
    span_tag = topic.find('span', class_='s')
    if span_tag:
        view_tag = span_tag.find_all('b')[2]
        if view_tag:
            views = view_tag.get_text()
    return views

def getTopicAuthor(topic):
    author = ''
    span_tag = topic.find('span', class_='s')
    if span_tag:
        author_tag = span_tag.find('a')
        if author_tag:
            author = author_tag.get_text()
    return author

def parse_topic(topic):
    """retrieve posts from topic"""
    page = 0 # start from the first page
    next_page = True
    index_post = ''
    previous_index_post = ''

    data = []

    # 
    while next_page:
        start_url = 'https://www.nairaland.com/{}/{}'.format(topic, page)
        r1 = requests.get(start_url, heads)
        topic_html = BeautifulSoup(r1.text, 'lxml')

        headers = topic_html.find_all('td', class_='bold l pu')
        bodys = topic_html.find_all('td', class_='l w pd')

        # retrieve first post in the topic
        if len(headers) > 1:
            index_post = getPostID(headers[0]) 

        # compare first post on current page with first post on previous page
        # to check if previous page and current page are the same
        if page > 0:
            if is_post_equal(index_post, previous_index_post):
                break

        for i in range(len(headers)):
            header = headers[i]
            body = bodys[i]
            post = parse_post(header, body)
            post.update({'page_no': page, 'topic':topic})
            data.append(post)
        
        previous_index_post = index_post
        page += 1
    print('topic: {}, No of Page(s): {}, No of Post(s) {}'.format(topic, page, len(data)))

    return data

def is_post_equal(post1, post2):
    return post1 == post2

def parse_post(header, body):
    """retrieve details of each post"""
    post = {}
    post['posted'] = getTimestamp(header)
    post['user'] = getUser(header)
    post['post_id'] = getPostID(header)
    post['text'] = getText(body)
    post['has_quote'] = True if getQuote(body) else False
    post['quotes'] = getQuote(body)
    post['shares'] = getLikesShares(body)[1]
    post['likes'] = getLikesShares(body)[0]
    post['retrieved'] = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    return post

def getUser(header):
    """Attempt to get user from post."""
    user = ''
    user_tag = header.find("a", class_="user")
    if user_tag:
        user = user_tag.get_text()
    return user

def getGender(header):
    """Attempt to get gender of user from post."""
    gender = ''
    female_tag = header.find("span", class_="f")
    male_tag = header.find("span", class_="m")
    if male_tag:
        gender = male_tag.get_text()
    elif female_tag:
        gender = female_tag.get_text()
    return gender

def getTimestamp(header):
    """Attempt to get timestamp of post."""
    time = ''
    date = ''
    tag_datetime = header.contents[-1]
    if tag_datetime:
        time = tag_datetime.contents[0].contents[0]
    if len(tag_datetime.contents) == 3:
        date = tag_datetime.contents[-1].contents[0]
    elif len(tag_datetime.contents) == 6:
        date = tag_datetime.contents[2].contents[0] + ' ' + tag_datetime.contents[4].contents[0]
    date_time = '{} {}'.format(date, time) 
    return date_time

def getLikesShares(body):
    """Attempt to get no of likes and shares."""
    likes = ''
    shares = ''
    s_class = body.find("p", class_='s')
    if s_class:
        likes = s_class.find_all('b')[0].get_text()
        shares = s_class.find_all('b')[1].get_text()
    return [likes, shares]

def getQuote(body):
    """Attempt to get quotes from post"""
    quotes = []
    content = body.find('div', class_='narrow')
    blockquotes = content.find_all('blockquote')
    for blockquote in blockquotes:
        a_tag = blockquote.find('a')
        if a_tag:
            _id = a_tag.get('href')
            quotes.append(_id)
    return quotes

def getText(body):
    """Attempt to get text from post"""
    content = body.find('div', class_='narrow')
    text = ''
    while content.blockquote:
        content.blockquote.extract()
    text = content.get_text()
    return text

def getPostID(header):
    """Attempt to get id of post"""
    post_id = ''
    name = header.find_all('a')[0].get('name')
    if name:
        post_id = name
    return post_id
    
if __name__ == '__main__':
    main()