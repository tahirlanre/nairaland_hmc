# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from tqdm import tqdm


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
