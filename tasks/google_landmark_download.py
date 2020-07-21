import sys, os, multiprocessing, csv
from urllib import request, error
from PIL import Image
from io import BytesIO
import pandas as pd

import tqdm


def parse_data(data_file):
    csvfile = pd.read_csv(data_file)

    return list(zip(csvfile["id"], csvfile["url"]))


def download_image(key_url):
    out_dir = ".././dataset/google"
    (key, url) = key_url
    filename = os.path.join(out_dir, '{}.jpg'.format(key))

    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0

    try:
        response = request.urlopen(url)
        image_data = response.read()
    except Exception as e:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1

    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1

    try:
        pil_image_rgb = pil_image.resize((512, 512)).convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=100)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1

    return 0


def loader():

    data_file = ".././dataset/csv/train.csv"
    key_url_list = parse_data(data_file)
    pool = multiprocessing.Pool(processes=1)  # Num of CPUs
    failures = sum(
        tqdm.tqdm(pool.imap_unordered(download_image, key_url_list),
                  total=len(key_url_list)))
    print('Total number of download failures:', failures)
    pool.close()
    pool.terminate()


if __name__ == '__main__':
    loader()
