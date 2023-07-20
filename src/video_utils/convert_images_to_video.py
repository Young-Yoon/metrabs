import os
import cv2
import argparse

def convert_images_to_video(image_directory, output_path, frame_rate=24, file_types=None, size=None, video_format='mp4'):
    # Mapping from video format to FOURCC code
    fourcc_map = {'mp4': 'mp4v', 'avi': 'MJPG'}

    if video_format not in fourcc_map:
        print(f'Unsupported video format: {video_format}')
        return

    fourcc_code = fourcc_map[video_format]

    # Get a list of image files in the directory
    image_files = sorted([f for f in os.listdir(image_directory) 
                          if os.path.isfile(os.path.join(image_directory, f)) 
                          and (os.path.splitext(f)[1].lower() in file_types)])

    # Check if there are any image files to process
    if not image_files:
        print('No image files found in the specified directory.')
        return

    # Initialize the video writer
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc_code), frame_rate, size)

    # Loop through the image files and write them to the video
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(image_path)

        # Resize image
        image = cv2.resize(image, size)

        video_writer.write(image)

    # Release the video writer
    video_writer.release()

    print("Image sequence converted to video.")

def main():
    parser = argparse.ArgumentParser(description='Convert image sequence to video')
    parser.add_argument('image_directory', type=str, help='Path to the image sequence directory')
    parser.add_argument('output_path', type=str, help='Path to save the output video')
    parser.add_argument('--frame_rate', type=int, default=24, help='Frame rate for the output video (default: 24)')
    parser.add_argument('--file_types', type=str, default='.jpg,.png', help='Comma-separated list of accepted file types (default: .jpg,.png)')
    parser.add_argument('--size', type=str, default='640x480', help='Desired output image size in format WIDTHxHEIGHT (default: 640x480)')
    parser.add_argument('--video_format', type=str, default='mp4', help='Video format for the output video (default: mp4)')
    args = parser.parse_args()

    # Convert the comma-separated string of file types to a list
    file_types = args.file_types.lower().split(',')

    # Convert the size argument to a tuple of integers
    size = tuple(map(int, args.size.lower().split('x')))

    # Get the video format
    video_format = args.video_format.lower()

    convert_images_to_video(args.image_directory, args.output_path, args.frame_rate, file_types, size, video_format)

if __name__ == "__main__":
    main()
