
# Approch 1 - Using Hough Circle
   Algorithms used to detect the image with circles and display its redius
# 1. Read the image path from the configuration file
# 2. Generate a list of contact size details for each pin
# 3. Load the image using OpenCV
# 4. Check if the image was loaded successfully
# 5. Convert the image to grayscale for processing
# 6. Apply GaussianBlur to reduce noise and improve circle detection
# 7. Use adaptive thresholding to create a binary image
# 8. Use Hough Circle Transform to detect circles in the binary image
            #circles = cv2.HoughCircles(
            #binary_image,
            # cv2.HOUGH_GRADIENT,
            # dp=1,  # Inverse ratio of the accumulator resolution to the image resolution
            # minDist=30,  # Minimum distance between detected centers
            # param1=100,  # Higher threshold for the Canny edge detector
            # param2=20,  # Accumulator threshold for the circle centers
            # minRadius=5,  # Minimum circle radius to be detected
            # maxRadius=50  # Maximum circle radius to be detected
            # )
# 9. Check if any circles were detected
# 10. Round the circle parameters to integers
# 11. Initialize total radius for average calculation
# 12. Loop through each detected circle
# 13. draw the detected circle on the image for visualization
# 14. Count the total number of detected circles
# 15. Calculate the average radius, ensuring no division by zero
# 16. Display the image with detected circles
# 17. wait for a key press to close the window
# 18. Close the image window
# 19. Return detected circle info


# Approch 2 - Using Random Forest Classifier
Importing Libraries:
    import os: Provides functions for interacting with the operating system.
    import cv2: OpenCV library for image processing.
    import pandas as pd: Pandas library for data manipulation.
    import numpy as np: NumPy library for numerical computations.
    from sklearn.ensemble import RandomForestClassifier: Random Forest classifier from scikit-learn.
    from sklearn.model_selection import RandomizedSearchCV: Randomized search for hyperparameter tuning.
    from scipy.stats import randint: Random integer generation from SciPy.

Data Preparation:
    The script defines some image details such as filenames, gauge sizes, and pin numbers.
    Data is prepared in a structured format using Pandas DataFrame.

Image Loading and Augmentation:
    Images are loaded using OpenCV (cv2.imread) and resized to a specified dimension.
    Data augmentation techniques like horizontal flipping and rotation are applied to increase the diversity of the training   dataset.

Model Training:
    Hyperparameters for Random Forest classifiers are defined.
    Randomized search (RandomizedSearchCV) is used to find the best hyperparameters.
    Separate classifiers are trained for predicting gauge sizes and pin numbers.
    The best models are fitted on the entire dataset, and predictions are made.
    Predictions are stored in a Pandas DataFrame and saved to a CSV file.


Return Statement:
        The function define_model() returns the best-trained classifiers for gauge sizes.


Scope:
    1. As data is less, we are getting 95.45 per accuracy.  I have also used gradient-descent and decison tress also but getting less accuracy
    2. I have not used Neural network problem as data is very less. 


Install:
pip install argparse
pip install pandas
pip install numpy
pip install scikit-learn
pip install scipy


Run command:
python <python file_name> --train_images_folder <train_image_dir> --test_images_folder <test_image_dir>
