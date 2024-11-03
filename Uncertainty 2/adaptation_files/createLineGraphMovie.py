import cv2
import os

# Set the directory containing images
image_folder = './line'
video_name = 'output_video.mp4'

# Get a list of all image files in the folder
# This will select only files that start with 'frame_' and end with '.png'
images = [img for img in os.listdir(image_folder) if img.startswith("frame_") and img.endswith(".png")]

# Sort the images based on the numerical part after 'frame_'
images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

# Read the first image to get the size (assuming all images are the same size)
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

# Loop through all images and write them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the video writer
video.release()

print("Video created successfully!")
