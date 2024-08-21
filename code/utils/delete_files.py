import os
from PIL import Image
import pandas as pd


def check_and_delete_invalid_images(folder_path, output_csv):
    invalid_images = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):  # Ensure you're only working with image files
            file_path = os.path.join(folder_path, filename)
            try:
                # Attempt to open the image
                with Image.open(file_path) as img:
                    img.verify()  # Verify that it is indeed an image
            except (IOError, SyntaxError) as e:
                # If an error occurs, log the file name and delete the file
                print(f"Removing invalid image: {file_path}, error: {e}")
                invalid_images.append(filename)
                os.remove(file_path)  # Delete the invalid image file

    # Save the invalid image names to a CSV file
    df = pd.DataFrame(invalid_images, columns=["Invalid Image"])
    df.to_csv(output_csv, index=False)
    print(f"Invalid images saved to {output_csv}")


# Path to the folder containing the images
folder_path = (
    "/data/talya/deep-learning-course-project/code/2019/ISIC_2019_Training_Input"
)

# Path to save the CSV file containing names of deleted images
output_csv = "/data/talya/deep-learning-course-project/code/2019/invalid_images.csv"

# Run the function
check_and_delete_invalid_images(folder_path, output_csv)
