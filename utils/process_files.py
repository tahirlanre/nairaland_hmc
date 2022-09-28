from time import time
from concurrent.futures import ProcessPoolExecutor
import itertools
import glob
import json
import gzip

import modin.pandas as pd
import ray
ray.init()

forum = "business"
fnames = glob.glob(f"data/raw/{forum}/*.gz")

print(f"Total no of files = {len(fnames)}")

def get_data(fname):
    data = []
    with gzip.open(fname, 'rb') as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data

ts = time()
with ProcessPoolExecutor() as executor:
    data = executor.map(get_data, fnames)

raw_data = pd.DataFrame(itertools.chain(*data))    
print(f"Took {time() - ts} seconds")

print("Saving to file...")
raw_data.to_json(f"data/{forum}_posts.json", lines=True, orient="records")

