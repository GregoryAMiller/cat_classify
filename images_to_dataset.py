import numpy as np
import pandas as pd
from PIL import Image
import os
from io import BytesIO
import pickle

# # Create an empty DataFrame to store the data
# df = pd.DataFrame()

# # Loop through all the image files in the folder
# for folder in os.listdir("C:/Users/Grego/Desktop/cat_classify/images/"):
#     for file in os.listdir(f"C:/Users/Grego/Desktop/cat_classify/images/{folder}"):
#         # Load the image file
#         image_path = f'C:/Users/Grego/Desktop/cat_classify/images/{folder}/{file}'
#         image = Image.open(image_path)
#         print(f'Loading {file}.......')

#         # Convert the PIL Image to numpy array and then to a comma-separated string
#         img_array = np.array(image)
#         img_string = ','.join(map(str, img_array.flatten()))

#         # Extract the label from the file name (assumes the file name is in the format "image_<label>.jpg")
#         label = int(file.split("_")[1].split(".")[0])

#         # Create a new row with the image string and label
#         row = {"image": img_string, "label": label}

#         # Concatenate the new row to the DataFrame
#         df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
  
# # Save the DataFrame to a CSV file
# print('Saving data to CSV file......')
# df.to_csv("C:/Users/Grego/Desktop/cat_classify/dataset.csv", index=False)

# List to store dataset entries
dataset_entries = []

# Loop through all the image files in the folder
base_path = "C:/Users/Grego/Desktop/cat_classify/images/"
for folder in os.listdir(base_path):
    for file in os.listdir(os.path.join(base_path, folder)):
        # Load the image file
        image_path = os.path.join(base_path, folder, file)
        image = Image.open(image_path)
        print(f'Loading {file}.......')

        # If you want to store the image as a numpy array:
        img_array = np.array(image)
        
        # Alternatively, if you want to store the image as bytes:
        # with open(image_path, 'rb') as f:
        #     img_bytes = f.read()

        # Extract the label from the file name (assumes the file name is in the format "image_<label>.jpg")
        label = int(file.split("_")[1].split(".")[0])

        # Create a new entry with the image data and label
        entry = {"image": img_array, "label": label}  # Use "image": img_bytes if using the bytes version

        # Append the new entry to the dataset entries list
        dataset_entries.append(entry)

with open('cat_dataset.pkl', 'wb') as f:
    pickle.dump(dataset_entries, f)