import argparse
import cv2
import os
import re

def resize_image(image, width=None, height=None, preserve_aspect_ratio=True):
    if width is None and height is None:
        return image
    if width is None or height is None:
        original_height, original_width = image.shape[:2]
        if preserve_aspect_ratio:
            if width is None:
                ratio = height / float(original_height)
                width = int(original_width * ratio)
            else:
                ratio = width / float(original_width)
                height = int(original_height * ratio)
        else:
            if width is None:
                width = original_width
            else:
                height = original_height
    return cv2.resize(image, (width, height))

def crop_image(image, x, y, width, height):
    return image[y:y+height, x:x+width]

def render_video(folder_path, output_filename, frame_rate, file_extension, output_format, width=None, height=None, preserve_aspect_ratio=True, crop_x=None, crop_y=None, crop_width=None, crop_height=None):
    # Get the list of file names in the folder with the specified extension
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(file_extension.lower())]

    # Sort the file names using regular expression and extract the file number
    file_names.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Check if there are enough images in the folder
    if len(file_names) == 0:
        raise ValueError("No image files found in the folder: {}".format(folder_path))

    # Get the first file to use as a template for the output video
    template_image = cv2.imread(os.path.join(folder_path, file_names[0]))

    # Resize the template image if width or height is specified
    template_image = resize_image(template_image, width, height, preserve_aspect_ratio)

    # Crop the template image if crop parameters are specified
    if crop_x is not None and crop_y is not None and crop_width is not None and crop_height is not None:
        template_image = crop_image(template_image, crop_x, crop_y, crop_width, crop_height)

    # Get the dimensions of the template image
    height, width, _ = template_image.shape

    # Determine the fourcc codec based on the output format
    if output_format.lower() == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif output_format.lower() == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        raise ValueError("Unsupported output format: {}".format(output_format))

    # Create a video writer object
    video_writer = cv2.VideoWriter(output_filename, fourcc, frame_rate, (width, height))

    # Loop through the file names and add them to the output video
    for file_name in file_names:
        # Read the image
        image = cv2.imread(os.path.join(folder_path, file_name))

        # Resize the image if width or height is specified
        image = resize_image(image, width, height, preserve_aspect_ratio)

        # Crop the image if crop parameters are specified
        if crop_x is not None and crop_y is not None and crop_width is not None and crop_height is not None:
            image = crop_image(image, crop_x, crop_y, crop_width, crop_height)

        # Add the image to the video
        video_writer.write(image)

    # Release the video writer object
    video_writer.release()

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Render a video from a sequence of images")

    # Add the command line arguments
    parser.add_argument("folder_path", help="Path to the folder containing the image sequence")
    parser.add_argument("output_filename", help="Name and path of the output video file")
    parser.add_argument("--frame_rate", type=int, default=30, help="Frame rate of the output video (default: 30)")
    parser.add_argument("--file_extension", default=".png", help="File extension of the image files to include (default: .png)")
    parser.add_argument("--output_format", default="mp4", choices=["mp4", "avi"], help="Output video format (default: mp4)")
    parser.add_argument("--width", type=int, help="Output video width")
    parser.add_argument("--height", type=int, help="Output video height")
    parser.add_argument("--preserve_aspect_ratio", action="store_true", help="Preserve the aspect ratio when resizing")
    parser.add_argument("--crop_x", type=int, help="X-coordinate of the top-left corner for cropping")
    parser.add_argument("--crop_y", type=int, help="Y-coordinate of the top-left corner for cropping")
    parser.add_argument("--crop_width", type=int, help="Width of the cropped region")
    parser.add_argument("--crop_height", type=int, help="Height of the cropped region")

    # Parse the command line arguments
    args = parser.parse_args()

    # Validate the folder path
    if not os.path.isdir(args.folder_path):
        raise ValueError("Invalid folder path: {}".format(args.folder_path))

    # Call the render_video function with the command line arguments
    render_video(
        args.folder_path,
        args.output_filename,
        args.frame_rate,
        args.file_extension,
        args.output_format,
        args.width,
        args.height,
        args.preserve_aspect_ratio,
        args.crop_x,
        args.crop_y,
        args.crop_width,
        args.crop_height
    )

if __name__ == "__main__":
    main()
