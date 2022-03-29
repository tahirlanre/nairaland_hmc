import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm
import csv
import json
from concurrent.futures import ThreadPoolExecutor

import os

# set headers
heads = requests.utils.default_headers()
heads.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
})
def is_post_equal(post1, post2):
    return post1 == post2

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
    text = re.sub("\\r", u"", text) # remove carriage returns
    return text

def getPostID(header):
    """Attempt to get id of post"""
    post_id = ''
    name = header.find_all('a')[0].get('name')
    if name:
        post_id = name
    return post_id

def parse_post(header, body):
    """retrieve details of each post"""
    post = {}
    post['posted'] = getTimestamp(header)
    post['user'] = getUser(header)
    post['gender'] = getGender(header)
    post['post_id'] = getPostID(header)
    post['text'] = getText(body)
    post['has_quote'] = True if getQuote(body) else False
    post['quotes'] = getQuote(body)
    post['shares'] = getLikesShares(body)[1]
    post['likes'] = getLikesShares(body)[0]
    post['retrieved'] = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    return post

# def parse_topic(topic):
#     """retrieve posts from topic"""
#     page = 0 # start from the first page
#     next_page = True
#     index_post = ''
#     previous_index_post = ''

#     data = []
#     # 
#     while next_page:
#         start_url = 'https://www.nairaland.com/{}/{}'.format(topic, page)
#         r1 = requests.get(start_url, heads)
#         topic_html = BeautifulSoup(r1.text, 'lxml')

#         headers = topic_html.find_all('td', class_='bold l pu')
#         bodys = topic_html.find_all('td', class_='l w pd')

#         # retrieve first post in the topic
#         if len(headers) > 1:
#             index_post = getPostID(headers[0]) 

#         # compare first post on current page with first post on previous page
#         # to check if previous page and current page are the same
#         if page > 0:
#             if is_post_equal(index_post, previous_index_post):
#                 break

#         for i in range(len(headers)):
#             header = headers[i]
#             body = bodys[i]
#             post = parse_post(header, body)
#             post.update({'page_no': page, 'topic':topic})
#             data.append(post)
        
#         previous_index_post = index_post
#         page += 1
#     print('topic: {}, No of Page(s): {}, No of Post(s) {}'.format(topic, page, len(data)))

#     return data

def parse_thread(thread):
    """retrieve posts from thread"""
    page = 0 # start from the first page
    next_page = True
    index_post = ''
    previous_index_post = ''

    data = []
    no_of_posts = 0

    headers =  ['post_id',
                'date',
                'user',
                'gender',
                'text',
                'has_quote',
                'quotes',
                'shares',
                'likes',
                'retrieved',
                'page_no',
                ]
    output_dir = './data/raw/'
    output_file = os.path.join(output_dir, f'{thread}.csv')
    if os.path.exists(output_file):
        raise ValueError('File exists')
    with open(output_file, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(headers)
    csvFile.close()

    print(f'****** Collecting posts from thread id: {thread} ******')

    while next_page:
        start_url = 'https://www.nairaland.com/{}/{}'.format(thread, page)
        r1 = requests.get(start_url, heads)
        thread_html = BeautifulSoup(r1.text, 'lxml')

        headers = thread_html.find_all('td', class_='bold l pu')
        bodys = thread_html.find_all('td', class_='l w pd')

        #retrieve first post in the thread
        if len(headers) > 1:
            index_post = getPostID(headers[0]) 

        if page > 0:
            # compare first post on current page with previous page
            if is_post_equal(index_post, previous_index_post):
                break

        for i in range(len(headers)):
            header = headers[i]
            body = bodys[i]
            # post = parse_post(header, body)
            # post.update({'page_no': page, 'thread':thread})
            # data.append(post)
            posted = getTimestamp(header)
            user = getUser(header)
            gender = getGender(header)
            post_id = getPostID(header)
            text = getText(body)
            has_quote = True if getQuote(body) else False
            quotes = getQuote(body)
            shares = getLikesShares(body)[1]
            likes = getLikesShares(body)[0]
            retrieved = datetime.now().strftime("%H:%M:%S %d-%m-%Y")

            rows = [
                post_id,
                posted,
                user,
                gender,
                text,
                has_quote,
                quotes,
                shares,
                likes,
                retrieved,
                page
                ]

            with open(output_file, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(rows)
            csvFile.close()

            no_of_posts += 1
        
        previous_index_post = index_post
        page += 1
    print('Thread: {}, No of Page(s): {}, No of Post(s) {}'.format(thread, page, no_of_posts))

if __name__ == "__main__":
    threads = []
    with open("data/health_topics.json", "r") as f:
        for line in f:
            obj = json.loads(line)
            threads.append(obj["topic_id"])

    # with ThreadPoolExecutor() as executor:
    #     executor.map(parse_thread, threads)

    # for thread in threads:
    #     parse_thread(thread)
        # df = pd.DataFrame(posts)
        # df.to_csv('data/raw/{}.csv'.format(thread), index=False)