import glob
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import umap.umap_ as umap
import rasterfairy
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='create an image grid sorted via umap dimention reduction.')
parser.add_argument('-i', '--input-dir', type=str, required=True, help='Directory of input images')
parser.add_argument('-o', '--output-image', type=str, default='out.png', help='Name of the output stitched image. Default is input_dir/out.png')
parser.add_argument('-w', '--image-width', type=int, required=True, help='Width of each input image')
parser.add_argument('-h', '--image-height', type=int, default=-1, help='Height of each input image. Default is image_width')
parser.add_argument('-g', '--grid-size', type=int, required=True, help='Size of the square output grid. If -r is provided, this only specifies the number of columns and the grid will be rectangular')
parser.add_argument('-r', '--grid-rows', type=int, default=-1, help='Number of rows in the output grid')

args = parser.parse_args()

# GET IMAGE EMBEDDINGS

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_embeddings(model, img_path):
    img_array = load_and_preprocess_image(img_path)
    embeddings = model.predict(img_array)
    return embeddings

image_files = glob.glob("./aligned_pennies/*.png")

def get_image_embeddings(image_files):

    # Load the InceptionV3 model without the top classification layer
    base_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')

    embeddings = []

    for img_path in image_files:
        if 'bad' in img_path:
            continue

        try:
            img_id = os.path.basename(img_path)
            e = extract_embeddings(base_model, img_path)
            embeddings.append([img_id] + e)
        except Exception as e:
            print(f"Error processing image {img_path}: \n{e}")

    # # Save embeddings to a CSV file
    # with open('image_embeddings_iV3_blackbg.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['image_id', 'embedding'])
    #     for img_id, embedding in embeddings_dict.items():
    #         writer.writerow([img_id, embedding.tolist()])

    return np.array(embeddings)

# GET UMAP DIMENTION REDUCTION

def umap_dimention_reduction(feature_data):

    reducer = umap.UMAP()
    # feature_data = pd.read_csv('image_embeddings_iV3_blackbg.csv')

    feature_headers = [f'em{i}' for i in range(2048)]

    features = feature_data[ feature_headers ]

    embedding = reducer.fit_transform(features)
    # print(embedding.shape)
    return embedding

    # Save the UMAP embeddings to a CSV file
    # with open('image_embeddings_iV3_umap_blackbg.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['image_id', 'embedding'])
    #     for i, e in enumerate(embedding):
    #         writer.writerow([i, e.tolist()])

    # plt.scatter(embedding[:, 0], embedding[:, 1])
    # plt.show()


# RECTANGLE GRID

def map(val, ilo, ihi, olo, ohi):
    return olo + (ohi - olo) * ((val - ilo) / (ihi - ilo))

def relp(pointcloud):

    pointcloud = np.array(pointcloud[['x', 'y']])
    
    # TODO: don't repeat this code
    grid_width = args.grid_size
    grid_height = args.grid_rows if args.grid_rows > 0 else args.grid_size
    n_points = grid_width * grid_height
    pointcloud = pointcloud[:n_points]

    grid = rasterfairy.transformPointCloud2D(pointcloud)

    grid = np.array(grid[0])

    return grid

def stitch_image(grid):

    # TODO: don't repeat this code
    grid_width = args.grid_size
    grid_height = args.grid_rows if args.grid_rows > 0 else args.grid_size
    n_points = grid_width * grid_height
    image_width = args.image_width
    image_height = args.image_height if args.image_height > 0 else args.image_width

    canvas_width = image_width * grid_width
    canvas_height = image_height * grid_height
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    input_directory = args.input_dir
    for i, (image_id, x, y) in enumerate(grid):
        image_id = int(image_id)
        x = int(x)
        y = int(y)

        image_path = os.path.join(input_directory, f'{image_id}.png')
        print(f'{round(100*(i / n_points), 1)}% : {image_path}')
        img = Image.open(image_path).resize((image_width, image_height))

        canvas.paste(img, (x * image_width, y * image_height))


    print("Saving the final image...")
    canvas.show()

    output_path = 'yearmap.png'
    canvas.save(output_path)
    print(f"Stitched image saved as {output_path}")

def main():
    embeddings = get_image_embeddings(args.input_dir)
    pointcloud = umap_dimention_reduction(embeddings)
    grid = relp(pointcloud)
    stitch_image(grid)