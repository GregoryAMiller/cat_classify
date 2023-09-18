import numpy as np
import pandas as pd
from PIL import Image
import os

# Create an empty DataFrame to store the data
df = pd.DataFrame()

# Loop through all the image files in the folder
for folder in os.listdir("C:/Users/Grego/Documents/chat_GPT/cat_classify/images/"):
  for file in os.listdir(f"C:/Users/Grego/Documents/chat_GPT/cat_classify/images/{folder}"):
    # Load the image file
    image = Image.open(f'C:/Users/Grego/Documents/chat_GPT/cat_classify/images/{folder}/{file}')
    print(f'Loading {file}.......')
  
    # Preprocess the image as needed (e.g. resize, convert to grayscale, normalize)
    # image = np.array(image) / 255
  
    # Flatten the image into a 1D array of pixels
    # pixels = image.flatten()
  
    # Extract the label from the file name (assumes the file name is in the format "image_<label>.jpg")
    label = int(file.split("_")[1].split(".")[0])
  
    # Create a new row with one column for each pixel
    row = {}
    # for i, pixel in enumerate(pixels):
    #   row["pixel_{}".format(i)] = pixel
    row["image"] = image
    row["label"] = label
  
    # Concatenate the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
  
# Save the DataFrame to a CSV file
print('Saving data to CSV file......')
df.to_csv("C:/Users/Grego/Documents/chat_GPT/cat_classify/dataset.csv", index=False)