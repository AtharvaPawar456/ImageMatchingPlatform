import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculateImageSimilarity(imagePath1: str, imagePath2: str) -> float:
    """
    Calculates the Structural Similarity Index (SSIM) between two images.
    
    Args:
        imagePath1 (str): Path to the first image.
        imagePath2 (str): Path to the second image.
    
    Returns:
        float: Similarity score between 0 and 1 (1 means identical).
    """
    try:
        # Load images in grayscale
        img1 = cv2.imread(imagePath1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)

        # Check if images are loaded properly
        if img1 is None or img2 is None:
            raise ValueError("One or both image paths are invalid or the images could not be loaded.")

        # Resize images to the same dimensions (if different)
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Compute SSIM
        similarity, _ = ssim(img1, img2, full=True)
        return similarity

    except Exception as e:
        print(f"Error: {e}")
        return 0.0

if __name__ == "__main__":
    image1Path = "rings/ring.jpg"
    image2Path = "rings/ring2.jpg"
    
    similarityScore = calculateImageSimilarity(image1Path, image2Path)
    print(f"Similarity Score: {similarityScore:.4f}")
