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
@click.argument('--forums')
def main(forums):
    """ Runs web scraping scripts to turn collect data from the site and
        copy into (../raw) 
    """
    for forum in forums:
        parse_forum(forum)

def parse_forum(forum):
    start_url = 'https://www.nairaland.com/{}/'.format(forum)
    r1 = requests.get(start_url, heads)
    raw_html = BeautifulSoup(r1.text, 'html5lib')

    # retrieve the number of pages in the forum
    page = int(raw_html.select('body > div > p:nth-child(7)')[0].select('b')[1].text) #
    
    for i in tqdm(range(page)):
        next_page = start_url + '{}'.format(i)
        r2 = requests.get(next_page, headers=headers)
        forum_html = BeautifulSoup(r2.text, 'html5lib')
        links = html_soup.find_all('a')
        for link in links:
            thread = get_thread(link)
            if thread:
                parse_thread(thread)

def get_thread(link):
    _id = ''
    name = post.get('name')
    if name not in ('top', None):
        _id = name
    return _id

# retrieve posts from thread
def parse_thread(thread):
    page = 0 # start from the first page
    next_page = True
    index_post = ''
    previous_index_post = ''

    data = []

    while next_page:
        start_url = 'https://www.nairaland.com/{}/{}'.format(thread, page)
        r1 = requests.get(start_url, heads)
        thread_html = BeautifulSoup(r1.text, 'lxml')

        headers = thread_html.find_all('td', class_='bold l pu')
        bodys = thread_html.find_all('td', class_='l w pd')

        #retrieve first post in the thread 
        index_post = getPostID(headers[0]) 

        if page > 1:
            # compare first post on current page with previous page
            if is_post_equal(index_post, previous_index_post):
                break

        for i in range(len(headers)):
            header = headers[i]
            body = bodys[i]
            post = parse_post(header, body)
            post.update({'page_no': page, 'forum': '', 'thread':thread})
            data.append(post)
        
        previous_index_post = index_post
        page += 1
    return data

def getPostID(header):
    post_id = ''
    name = header.find_all('a')[0].get('name')
    if name:
        post_id = name
    return post_id

def is_post_equal(post1, post2):
    return post1 == post2

def parse_post(header, body):
    post = {}
    post['posted'] = getTimestamp(header)
    post['retrieved'] = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    post['shares'] = getLikesShares(body)[1]
    post['likes'] = getLikesShares(body)[0]
    post['user'] = getUser(header)
    post['has_quote'] = True if getQuote(body) else False
    post['quotes'] = getQuote(body)
    post['text'] = getText(body)
    post['post_id'] = getPostID(header)
    return post

def getUser(header):
    user = ''
    user_tag = header.find("a", class_="user")
    if user_tag:
        user_tag.get_text()
    return user

def getTimestamp(header):
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
    likes = ''
    shares = ''
    s_class = body.find("p", class_='s')
    if s_class:
        likes = s_class.find_all('b')[0].get_text()
        shares = s_class.find_all('b')[1].get_text()
    return [likes, shares]

def getQuote(body):
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
    content = body.find('div', class_='narrow')
    quotes = getQuote(body)
    text = ''
    for i in range(0, len(quotes)):
        content.blockquote.extract()
    text = content.get_text()
    return text

def getPostID(header):
    post_id = ''
    name = header.find_all('a')[0].get('name')
    if name:
        post_id = name
    return post_id
    
if __name__ == '__main__':
    main()