import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm
import click

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
    r1 = requests.get(start_url, headers)
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

def parse_thread(thread):
    page = 0 #set page no to 0 to start from the first page
    next_page = True

    while next_page:
        start_url = 'https://www.nairaland.com/{}/{}'.format(thread, page)
        r1 = requests.get(start_url, headers)
        thread_html = BeautifulSoup(r1.text, 'lxml')

        headers = thread_html.find_all('td', class_='bold l pu')
        bodys = thread_html.find_all('td', class_='l w pd')

        for i in range(len(headers)):
            header = headers[i]
            body = bodys[i]
            parse_post(header, body)

def parse_post(header, body):
    user

if __name__ == '__main__':
    main()