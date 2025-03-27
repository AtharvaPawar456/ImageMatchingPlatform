import cv2
import numpy as np

def calculateImageFeatureSimilarity(imagePath1: str, imagePath2: str) -> float:
    """
    Computes the similarity between two images based on ORB feature matching.
    
    Args:
        imagePath1 (str): Path to the first image.
        imagePath2 (str): Path to the second image.
    
    Returns:
        float: Similarity score (higher means more similar).
    """
    try:
        # Load images in grayscale
        img1 = cv2.imread(imagePath1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)

        # Check if images are loaded properly
        if img1 is None or img2 is None:
            raise ValueError("One or both image paths are invalid or the images could not be loaded.")

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # Check if descriptors are found
        if descriptors1 is None or descriptors2 is None:
            raise ValueError("Could not compute descriptors for one or both images.")

        # Create a brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches based on distance (lower distance = better match)
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate similarity score (normalized to 0-1 range)
        numKeypoints = max(len(keypoints1), len(keypoints2))
        similarity = len(matches) / numKeypoints if numKeypoints > 0 else 0

        return similarity

    except Exception as e:
        print(f"Error: {e}")
        return 0.0

if __name__ == "__main__":
    image1Path = "rings/ring.jpg"
    
    image2Path = "rings/ring2.jpg"
    image2Path = "rings/ring3.jpg"
    image2Path = "rings/ring4.jpg"
    image2Path = "rings/ring5.jpg"
    
    similarityScore = calculateImageFeatureSimilarity(image1Path, image2Path)
    print(f"Feature-Based Similarity Score: {similarityScore:.4f}")
