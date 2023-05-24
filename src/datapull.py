import os
import pandas as pd
import json
import requests
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, required=True, help="Name of the CSV file containing the image URLs")
    parser.add_argument('-u', '--url_col', type=str, required=True, help="Name of the column containing the image URLs")
    parser.add_argument('-l', '--label_col', type=str, required=True, help="Name of the column containing the image labels")
    return parser.parse_args()

def load_data(data, url_col='url', label_col='index'):

    data = pd.read_csv(data)

    # Keep only those rows, whos url are strings
    data = data[data['url'].apply(lambda x: isinstance(x, str))]

    # create a directory for the images if it doesn't already exist
    if not os.path.exists('images'):
        os.makedirs('images')

    # create a dictionary to hold the image paths and labels
    image_data = {}

    # loop through the URLs and download the images
    for i, url in enumerate(data[url_col]):
        try:
            response = requests.get(url, stream=True)
            img = Image.open(response.raw)
            # resize the image
            img = img.resize((224, 224))
            # save the image to the "images" directory with a unique filename
            filename = f"image_{i}.jpg"
            img.save(os.path.join('images', filename), "JPEG")
            # add the image path and label to the dictionary
            image_data[filename] = {"label": data.iloc[i][label_col], "path": os.path.abspath(os.path.join('images', filename))}
        except Exception as e:
            print(f"Error downloading image at {url}: {e}")

    # write the dictionary to a JSON file
    with open('image_data.json', 'w') as f:
        json.dump(image_data, f)

    print("Jobs done!")

def main():
    args = parse_args()
    load_data(args.data, args.url_col, args.label_col)

if __name__== "__main__":
    main()