import moviepy.editor as mp
from PIL import Image
import pandas as pd
import os

# The frame rate of the video is 30 fps used for calculating when to save an image
frame_rate = 30

# Loop through all the video files in the folder
for file in os.listdir("C:/Users/Grego/Documents/chat_GPT/cat_classify/videos/Asha/"):
  # Load the image file
#   print(file)
    video = mp.VideoFileClip(f"C:/Users/Grego/Documents/chat_GPT/cat_classify/videos/Asha/{file}")
    print(f'Loading {file}.......')
    
    image_label = file.split(".")[0]

    # Iterate through the frames of the video
    for i, frame in enumerate(video.iter_frames()):
        # Save every other frame as a JPEG image
        if i % frame_rate == 0:
            # Preprocess the image as needed (e.g. resize, convert to grayscale)
            image = Image.fromarray(frame)
            image = image.resize((200, 200))
            # image = image.convert("L")
            image.save("C:/Users/Grego/Documents/chat_GPT/cat_classify/images/Asha/{}Z{}_1.jpg".format(image_label, i))

# Loop through all the video files in the folder
for file in os.listdir("C:/Users/Grego/Documents/chat_GPT/cat_classify/videos/Annie/"):
  # Load the image file
#   print(file)
    video = mp.VideoFileClip(f"C:/Users/Grego/Documents/chat_GPT/cat_classify/videos/Annie/{file}")
    print(f'Loading {file}.......')
    
    image_label = file.split(".")[0]

    # Iterate through the frames of the video
    for i, frame in enumerate(video.iter_frames()):
        # Save every other frame as a JPEG image
        if i % frame_rate == 0:
            # Preprocess the image as needed (e.g. resize, convert to grayscale)
            image = Image.fromarray(frame)
            image = image.resize((200, 200))
            # image = image.convert("L")
            image.save("C:/Users/Grego/Documents/chat_GPT/cat_classify/images/Annie/{}Z{}_3.jpg".format(image_label, i))


# Loop through all the video files in the folder
for file in os.listdir("C:/Users/Grego/Documents/chat_GPT/cat_classify/videos/Bree/"):
  # Load the image file
#   print(file)
    video = mp.VideoFileClip(f"C:/Users/Grego/Documents/chat_GPT/cat_classify/videos/Bree/{file}")
    print(f'Loading {file}.......')
    
    image_label = file.split(".")[0]

    # Iterate through the frames of the video
    for i, frame in enumerate(video.iter_frames()):
        # Save every other frame as a JPEG image
        if i % frame_rate == 0:
            # Preprocess the image as needed (e.g. resize, convert to grayscale)
            image = Image.fromarray(frame)
            image = image.resize((200, 200))
            # image = image.convert("L")
            image.save("C:/Users/Grego/Documents/chat_GPT/cat_classify/images/Bree/{}Z{}_2.jpg".format(image_label, i))
  
  
    
