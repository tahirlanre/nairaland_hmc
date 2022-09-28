import logging
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
import gzip
import os
from functools import partial

from utils.utils import init_logger

init_logger()

# set headers
heads = requests.utils.default_headers()
heads.update(
    {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0",
    }
)


def is_post_equal(post1, post2):
    return post1 == post2


def get_user(header):
    """Attempt to get user from post."""
    user = ""
    user_tag = header.find("a", class_="user")
    if user_tag:
        user = user_tag.get_text()
    return user


def get_gender(header):
    """Attempt to get gender of user from post."""
    gender = ""
    female_tag = header.find("span", class_="f")
    male_tag = header.find("span", class_="m")
    if male_tag:
        gender = male_tag.get_text()
    elif female_tag:
        gender = female_tag.get_text()
    return gender


def get_timestamp(header):
    """Attempt to get timestamp of post."""
    time = ""
    date = ""
    tag_datetime = header.contents[-1]
    if tag_datetime:
        time = tag_datetime.contents[0].contents[0]
    if len(tag_datetime.contents) == 3:
        date = tag_datetime.contents[-1].contents[0]
    elif len(tag_datetime.contents) == 6:
        date = (
            tag_datetime.contents[2].contents[0]
            + " "
            + tag_datetime.contents[4].contents[0]
        )
    date_time = "{} {}".format(date, time)
    return date_time


def get_likes_and_shares(body):
    """Attempt to get no of likes and shares."""
    likes = ""
    shares = ""
    s_class = body.find("p", class_="s")
    if s_class:
        likes = s_class.find_all("b")[0].get_text()
        shares = s_class.find_all("b")[1].get_text()
    return [likes, shares]


def get_quote(body):
    """Attempt to get quotes from post"""
    quotes = []
    content = body.find("div", class_="narrow")
    blockquotes = content.find_all("blockquote")
    for blockquote in blockquotes:
        a_tag = blockquote.find("a")
        if a_tag:
            _id = a_tag.get("href")
            quotes.append(_id)
    return quotes


def get_text(body):
    """Attempt to get text from post"""
    content = body.find("div", class_="narrow")
    text = ""
    while content.blockquote:
        content.blockquote.extract()
    text = content.get_text()
    text = re.sub("\\r", "", text)  # remove carriage returns
    return text


def get_post_id(header):
    """Attempt to get id of post"""
    post_id = ""
    name = header.find_all("a")[0].get("name")
    if name:
        post_id = name
    return post_id


def get_post_info(header, body):
    """retrieve details of each post"""
    post = {}
    post["posted"] = get_timestamp(header)
    post["user"] = get_user(header)
    post["gender"] = get_gender(header)
    post["post_id"] = get_post_id(header)
    post["text"] = get_text(body)
    post["has_quote"] = True if get_quote(body) else False
    post["quotes"] = get_quote(body)
    post["shares"] = get_likes_and_shares(body)[1]
    post["likes"] = get_likes_and_shares(body)[0]
    post["retrieved"] = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
    return post


def parse_thread(output_dir, thread):
    """retrieve posts from thread"""
    page = 0  # start from the first page
    next_page = True
    index_post = ""
    previous_index_post = ""

    data = []
    no_of_posts = 0
    output_file = os.path.join(output_dir, f"{thread}.jsonl.gz")
    if os.path.exists(output_file):
        raise ValueError("File exists")

    logging.info(f"****** collecting posts from thread id: {thread} ******")

    with gzip.open(output_file, "w") as output:
        while next_page:
            start_url = "https://www.nairaland.com/{}/{}".format(thread, page)
            r1 = requests.get(start_url, heads)
            thread_html = BeautifulSoup(r1.text, "lxml")

            headers = thread_html.find_all("td", class_="bold l pu")
            bodys = thread_html.find_all("td", class_="l w pd")

            # retrieve first post in the thread
            if len(headers) > 1:
                index_post = get_post_id(headers[0])

            if page > 0:
                # compare first post on current page with previous page
                if is_post_equal(index_post, previous_index_post):
                    break

            for i in range(len(headers)):
                obj = get_post_info(headers[i], bodys[i])
                output.write(json.dumps(obj).encode("utf-8") + b"\n")
                no_of_posts += 1

            previous_index_post = index_post
            page += 1
        logging.info(
            "post ID: {}, num of page(s): {}, num of post(s) {}".format(
                thread, page, no_of_posts
            )
        )


if __name__ == "__main__":
    threads = []
    forum = "politics"
    output_dir = f"/media/zqxh49/C28AAF378AAF273F/PHD/data/Nairaland/raw/{forum}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(f"data/{forum}_topics.json", "r") as f:
        for line in f:
            obj = json.loads(line)
            threads.append(obj["topic_id"])

    with ThreadPoolExecutor() as executor:
        fn = partial(parse_thread, output_dir)
        executor.map(fn, threads)
