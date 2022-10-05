from time import time
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import gzip

forum = "family"
fnames = glob.glob(
    f"/media/zqxh49/C28AAF378AAF273F/PHD/data/Nairaland/raw/{forum}/*.gz"
)

print(f"Total no of files = {len(fnames)}")


def get_data(fname):
    data = []
    with gzip.open(fname, "rb") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj)
    return data


ts = time()
with ProcessPoolExecutor() as executor:
    data = executor.map(get_data, fnames)

print(f"Took {time() - ts} seconds")

print("Saving to file...")
with open(f"data/{forum}_posts.json", "w") as f_w:
    for d in data:
        for item in d:
            f_w.write(json.dumps(item) + "\n")

# raw_data = pd.DataFrame(itertools.chain(*data))

# raw_data.to_json(f"data/{forum}_posts.json", lines=True, orient="records")
