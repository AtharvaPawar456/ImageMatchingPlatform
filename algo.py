import cv2
import numpy as np
import os
import json
from typing import List, Dict

def calculateImageFeatureSimilarity(imagePath1: str, imagePath2: str) -> float:
    """
    Computes the similarity between two images based on ORB feature matching.
    
    Args:
        imagePath1 (str): Path to the first image (search image).
        imagePath2 (str): Path to the second image (database image).
    
    Returns:
        float: Similarity score (higher means more similar).
    """
    try:
        # Load images in grayscale
        img1 = cv2.imread(imagePath1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(imagePath2, cv2.IMREAD_GRAYSCALE)

        # Check if images are loaded properly
        if img1 is None or img2 is None:
            raise ValueError(f"Error loading images: {imagePath1} or {imagePath2}")

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # Ensure descriptors exist
        if descriptors1 is None or descriptors2 is None:
            return 0.0  # No descriptors = no similarity

        # Create Brute Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)

        # Sort matches by distance (lower distance = better match)
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate similarity score (normalized)
        numKeypoints = max(len(keypoints1), len(keypoints2))
        similarity = len(matches) / numKeypoints if numKeypoints > 0 else 0.0

        return similarity

    except Exception as e:
        print(f"Error comparing images: {e}")
        return 0.0

def findMostSimilarImages(searchImagePath: str, databaseFolder: str, outputJson: str) -> List[Dict[str, float]]:
    """
    Compares a search image with all images in a folder and returns similarity scores.
    
    Args:
        searchImagePath (str): Path of the image to search.
        databaseFolder (str): Folder containing images to compare.
        outputJson (str): Path to save the JSON result.

    Returns:
        List[Dict[str, float]]: List of images with similarity scores, sorted by most similar.
    """
    try:
        imageScores = []

        # Loop through all images in the database folder
        for imgFile in os.listdir(databaseFolder):
            imgPath = os.path.join(databaseFolder, imgFile)

            # Check if it's an image
            if imgFile.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                similarity = calculateImageFeatureSimilarity(searchImagePath, imgPath)
                imageScores.append({"imgPath": imgPath, "similarityScore": similarity})

        # Sort images by similarity score (highest first)
        imageScores.sort(key=lambda x: x["similarityScore"], reverse=True)

        # Save results to JSON
        with open(outputJson, "w") as jsonFile:
            json.dump(imageScores, jsonFile, indent=4)

        return imageScores

    except Exception as e:
        print(f"Error finding similar images: {e}")
        return []

if __name__ == "__main__":
    searchImagePath = "archive/Jewellery_Data/necklace_43.jpg"  # Set your input image path
    databaseFolder = "archive/Jewellery_Data/necklace"  # Folder containing images
    outputJson = "similar_images.json"

    # Find and sort similar images
    sortedImages = findMostSimilarImages(searchImagePath, databaseFolder, outputJson)

    # Print sorted list
    print("\n🔍 Sorted Similar Images:")
    for img in sortedImages:
        print(f"{img['imgPath']} - Similarity: {img['similarityScore']:.4f}")

    # Print top 3 matches
    print("\n🏆 Top 3 Similar Images:")
    for topImage in sortedImages[:3]:
        print(f"{topImage['imgPath']} - Similarity: {topImage['similarityScore']:.4f}")
