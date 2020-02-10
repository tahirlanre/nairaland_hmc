import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm

def get_forum_posts(start_url):
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })

    start_url = start_url
    response = requests.get(start_url, headers)
    html = BeautifulSoup(response.text, 'html5lib')
    forum_posts = []

    page = int(html.select('body > div > p:nth-child(7)')[0].select('b')[1].text)

    for i in tqdm(range(page)):
        forum_url = start_url + '{}'.format(i)
        r = requests.get(forum_url, headers=headers)
        html_soup = BeautifulSoup(r.text, 'html5lib')
        n = 2
        while True:
            topic = html_soup.select("body > div > table:nth-child(9) > tbody > tr:nth-child({})".format(n))
            if len(topic) > 0:
                name = topic[0].select('a')[1].getText() #Get topic of post
                views = 0
                sub_url = topic[0].select('a')[1].get('href')
                no_of_posts = 0
                created_by = ''
                num_of_page = topic[0].select('a')[-3].text
                num_of_page = int(num_of_page[1:-1]) #remove brackets and convert to integer
                file_location = ''
                forum_posts.append([name, views, sub_url, no_of_posts, created_by, num_of_page, file_location])
            else:
                break
            n = n + 1
    return forum_posts

def getPostId(link):
    id = ''
    name = link.get('name')
    if name not in ('top',None):
        id = name
    return id

def get_forum_post_id(forum):
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })
    start_url = 'https://www.nairaland.com/{}/'.format(forum)
    response = requests.get(start_url, headers)
    html = BeautifulSoup(response.text, 'html5lib')
    post_ids = []

    page = int(html.select('body > div > p:nth-child(7)')[0].select('b')[1].text) #

    for i in tqdm(range(page)):
        forum_url = start_url + '{}'.format(i)
        r = requests.get(forum_url, headers=headers)
        html_soup = BeautifulSoup(r.text, 'html5lib')
        links = html_soup.find_all('a')
        for link in links:
            id = getPostId(link)
            if id:
                post_ids.append([id])
    return post_ids

forum = 'health'
result = get_forum_post_id(forum)
df = pd.DataFrame(result, columns=['id'])
df.to_csv('data/raw/{}.csv'.format(forum), index=False)
