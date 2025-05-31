import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import pandas as pd


class ImageLoad:
    def __init__(self, dataset_path: str, save_new_dataset: bool = 0, resize_to=(128, 128)):
        """
        Initializes the ImageLoad class.

        Args:
            dataset_path (str): Path to the dataset directory containing images.
            resize_to (tuple): Dimensions to resize images to (default: (64, 64)).
        """
        self.save_new_dataset = save_new_dataset
        self.size = resize_to
        self.dataset_path = dataset_path
        self.image_np_arrays = list() # populated in main_loop
        self.output_dir = 'data/output'
        self.percent = 0.80
        os.makedirs(self.output_dir, exist_ok=True)

        # Load image paths
        # dataframe is populated in main_loop
        self.df = pd.DataFrame()
        
    

    def main_loop(self) -> dict:
        """
        Processes all images and stores the results in a DataFrame.

        Returns:
            dict: A dictionary with processed images categorized by their labels.
        """
        print('Start image preprocessing...')
    
        processed_images_list = []  # To store the processed images for DataFrame

        for idx, (fname, img_path, category) in enumerate(tqdm(self.image_paths, desc="Processing images")):
        
            try:
                cat = True #if category == 'Benign' else False

                image = self._open_img(img_path, cat=cat)
                processed_images = [
                    image,
                    self._add_gaussian_blurr(image,cat),
                    self._add_gaussian_noise(image, cat),
                    self._flip_hort(image, cat),
                    self._flip_vert(image, cat),
                    self._rotate_90_clockwise(image, cat),
                    self._rotate_90_counter_clockwise(image, cat)
                ]

                processed_images = [x for x in processed_images if x is not None]
                self.image_np_arrays.append((category, processed_images))

                if self.save_new_dataset == 1:
                    for idx_i, img in enumerate(processed_images):
                        id_ = f"{idx + 1}_{idx_i}"
                        self.save_to_folders(fname, img, id_, category)

                processed_images_list.extend((category, img) for img in processed_images)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Parse each image into DataFrame as columns: (Label, Numpy Array)
        self.df = pd.DataFrame(processed_images_list, columns=['Label', 'Image'])
        
        # Find the minimum row count across all categories
        min_count = self.df['Label'].value_counts().min()

        # Sample each category to have the same number of rows as `min_count`
        self.df = self.df.groupby('Label').apply(lambda x: x.sample(n=min_count)).reset_index(drop=True)
        print('💖Preprocessing completed.💖')



    def _open_img(self, image_path: str, add_noise:bool= True, cat=False, show:bool=False) -> np.ndarray:
        """Opens and transforms the image into NumPy array form."""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        new_height = height - (height // 8) # crop out stamp on btm right
        img_resized = cv2.resize(img, self.size)  # Resize the image
        if show:
            cv2.imshow("Resized Image", img_resized)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()  # Close the window after key press

        img_resized = img_resized[0:new_height, 0:width]


        # # TODO add noise to each image
        # if add_noise and random.randint(0, 1) > 0.9 and cat == False:
        #     return self._add_gaussian_noise(img_resized)
        return img_resized

    def _add_gaussian_noise(self, image: np.ndarray, cat:bool=True, mean: float = 0, sigma: float = 5) -> np.ndarray:
        """Adds Gaussian noise to the image."""
        if random.randint(0, 1) > self.percent or cat == True:

            noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
            # Add the noise to the image
            return cv2.add(image, noise)
        return None

    def save_to_folders(self, fname: str, image: np.ndarray, idx: str, category: str):
        """Saves the processed image to the specified category folder."""
        # Ensure image is in the correct format
        if not isinstance(image, np.ndarray):
            raise ValueError("The image must be a NumPy array.")

        # Create the category folder if it doesn't exist
        category_path = os.path.join(self.output_dir, 'processed', category)
        os.makedirs(category_path, exist_ok=True)

        # Construct the full file path
        file_path = os.path.join(category_path, f"{idx}_{fname}")
        print(f"Saving image to: {file_path}")

        # Save the image
        success = cv2.imwrite(file_path, image)
        if not success:
            raise IOError(f"Failed to save the image at {file_path}")

    def _rotate_90_counter_clockwise(self, image: np.ndarray, cat:bool = False) -> np.ndarray:
        """Rotates the image 90 degrees counterclockwise."""
        if random.randint(0, 1) > self.percent or cat == True:
            return np.rot90(image, k=1)
        return None

    def _rotate_90_clockwise(self, image: np.ndarray, cat:bool = False) -> np.ndarray:
        """Rotates the image 90 degrees clockwise."""
        if random.randint(0, 1) > self.percent or cat == True:
            return np.rot90(image, k=-1)
        return None

    def _flip_hort(self, image: np.ndarray, cat:bool = False) -> np.ndarray:
        """flip over horzontal axis."""
        if random.randint(0, 1) > self.percent or cat == True:
            return cv2.flip(image, -1)
        return None
    
    def _flip_vert(self, image: np.ndarray, cat:bool = False) -> np.ndarray:
        """flip over vertical axis."""
        if random.randint(0, 1) > self.percent or cat == True:
            return cv2.flip(image, 1)
        return None

    def _add_gaussian_blurr(self, image: np.ndarray, cat:bool = False) -> np.ndarray:
        """Applies Gaussian blur to the image."""
        if random.randint(0, 1) > self.percent or cat == False:
            return cv2.GaussianBlur(image, (5, 5), 0)
        return None
    
    def _load_image_paths(self) -> list[tuple[str, str, str]]:
        """
        Recursively loads image file paths from the dataset directory.

        Returns:
            list[tuple[str, str, str]]: List of image file paths with their respective category.
        """
        image_extensions = ['.tiff', '.tif', '.jpg', '.jpeg', '.png']
        image_paths = []

        # Loop through the main dataset directory
        for category in os.listdir(self.dataset_path):
            category_path = os.path.join(self.dataset_path, category)
            if os.path.isdir(category_path):
                for fname in os.listdir(category_path):
                    if any(fname.lower().endswith(ext) for ext in image_extensions):
                        image_paths.append(
                            (fname, os.path.join(category_path, fname), category))

        return image_paths
    
   