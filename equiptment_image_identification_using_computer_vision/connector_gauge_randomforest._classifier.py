import os
import cv2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint
import argparse

def define_model(X, y_pins, image_names):
    # Define the parameter distributions for pin numbers
    param_dist_pins = {
        'n_estimators': randint(100, 300),
        'max_depth': [None] + list(randint(1, 100).rvs(5)),
        'min_samples_split': randint(2, 11)
    }

    # Initialize the Random Forest classifier for pin numbers
    clf_pins = RandomForestClassifier()

    # Initialize RandomizedSearchCV for pin numbers
    random_search_pins = RandomizedSearchCV(clf_pins, param_distributions=param_dist_pins, n_iter=30, cv=2,
                                            scoring='accuracy', random_state=42)

    # Perform random search for pin numbers
    random_search_pins.fit(X, y_pins)

    # Get the best estimator for pin numbers from random search
    best_estimator_pins = random_search_pins.best_estimator_

    # Fit the best estimator for pin numbers on the entire dataset
    best_estimator_pins.fit(X, y_pins)

    # Use the best estimator for pin numbers to make predictions
    y_pins_pred = best_estimator_pins.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y_pins, y_pins_pred)
    print("Accuracy for total connector identification:", accuracy)

    # Output the predictions
    predictions_df = pd.DataFrame({
        'Image_Name': image_names,  # Include image names in the output
        'Actual_Pin_Number': y_pins,
        'Predicted_Pin_Number': y_pins_pred
    })

    # Remove duplicates based on 'Image_Name'
    predictions_df = predictions_df.drop_duplicates(subset=['Image_Name'])

    predictions_df.to_csv('connector_predictions.csv', index=False)

    return best_estimator_pins


def train(train_images_folder):
    # Define the image details
    image_names = ['cconn_38999_9-5.png', 'cconn_38999_9-98.png', 'cconn_38999_11-98.png', 'cconn_38999_17-6',
                   'cconn_38999_21-16.png', 'cconn_cc8-98_t.png', 'cconn_cc12-3.png', 'cconn_cc14-4_t.png',
                   'cconn_cc14-7_t.png', 'cconn_cc16-10_t.png', 'cconn_cc16-10_t.png', 'cconn_cc18-8_t.png',
                   'rect_atm_ia_12.png', 'rect_bacc65_ia_1.png', 'rect_bacc65_ia_2_t.png', 'rect_bacc65_ia_3_t.png',
                   'rect_bacc65_ia_7.png', 'rect_ia_10ap_01w1_t.png', 'rect_ia_10ap_04_t.png', 'rect_ia_10ap_06f.png',
                   'rect_ia_10ap_06f.png', 'rect_ia_bacc65av2.png', 'rect_ia_bacc65aw1_t.png', 'rect_ia_bacc65aw2.png']
    pin_numbers = [[4], [3], [6], [6], [16], [3], [3], [4], [7], [10], [8], [12], [8], [8], [12], [24], [1], [4], [6],
                   [2], [2], [1], [2]]

    # Prepare the data
    data = []
    for image, pins in zip(image_names, pin_numbers):
        for pin in pins:
            data.append({'image_name': image, 'pin_number': pin})
    details_df = pd.DataFrame(data)

    # Load images
    def load_images(images_folder):
        images = {}
        for filename in os.listdir(images_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(images_folder, filename)
                image = cv2.imread(image_path)
                # Resize images if needed
                resized_img = cv2.resize(image, (50, 50))  # Adjust dimensions as needed
                images[filename] = resized_img
        return images

    # Data augmentation function using OpenCV
    def augment_image(image):
        augmented_images = [image]  # Initialize with original image
        # Horizontal flip
        augmented_images.append(cv2.flip(image, 1))
        # Rotation
        rows, cols, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 3, rows / 2), 10, 2)  # Rotate by 30 degrees
        augmented_images.append(cv2.warpAffine(image, rotation_matrix, (cols, rows)))
        return augmented_images

    # Preprocess image details and prepare data for classification
    def prepare_data(image_details_df, images):
        X = []
        y_pins = []
        image_names_expanded = []  # To keep track of image names for each augmented image
        for index, row in image_details_df.iterrows():
            image_name = row['image_name']
            pin_number = row['pin_number']
            if image_name in images:
                original_image = images[image_name]
                augmented_images = augment_image(original_image)
                for image in augmented_images:
                    # Flatten the image and add it to X
                    flattened_img = image.flatten()
                    X.append(flattened_img)
                    y_pins.append(pin_number)
                    image_names_expanded.append(image_name)  # Add the image name for each augmented image
        return np.array(X), np.array(y_pins), image_names_expanded

    # Load images
    images = load_images(train_images_folder)

    # Prepare data with data augmentation
    X, y_pins, image_names_expanded = prepare_data(details_df, images)

    best_estimator_pins = define_model(X, y_pins, image_names_expanded)

    return best_estimator_pins


def test(test_images_folder, best_estimator_pins):
    def load_test_images(images_folder):
        images = {}
        for filename in os.listdir(images_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                image_path = os.path.join(images_folder, filename)
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        images[filename] = image
                    else:
                        print(f"Error loading image: {image_path}")
                else:
                    print(f"File not found: {image_path}")
        return images

    def preprocess_image(image):
        # Check if the image is not empty
        if not image.size:
            print("Empty image encountered")
            return None
        # Resize image if needed
        resized_img = cv2.resize(image, (50, 50))  # Adjust dimensions as needed
        # Flatten the image
        flattened_img = resized_img.flatten()
        return flattened_img

    def predict_on_test_images(images, model_pins):
        predictions = {}
        for filename, image in images.items():
            # Preprocess the image
            preprocessed_image = preprocess_image(image)
            if preprocessed_image is None:
                continue
            # Use the trained model to predict pin number on the preprocessed image
            predicted_pin_number = model_pins.predict([preprocessed_image])[0]

            # Store the prediction for the image
            predictions[filename] = predicted_pin_number

        # Output the predictions
        printed_images = set()  # To keep track of printed images
        for filename, predicted_pin_number in predictions.items():
            if filename not in printed_images:
                #print(f"Predicted total connector number for {filename}: {predicted_pin_number}")
                printed_images.add(filename)  # Mark this image as printed

        return predictions

    # Load test images
    test_images = load_test_images(test_images_folder)

    # Use the trained models to predict on the test images
    predictions = predict_on_test_images(test_images, best_estimator_pins)

    # Output the predictions
    for filename, result in predictions.items():
        print(f"Predicted connector number for {filename}: {result}")


def main(train_images_folder, test_images_folder):
    best_estimator_pins = train(train_images_folder)
    test(test_images_folder, best_estimator_pins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_images_folder", required=True, help="Provide full path to the training images folder")
    parser.add_argument("--test_images_folder", required=True, help="Provide full path to the test images folder")
    args = parser.parse_args()
    main(args.train_images_folder, args.test_images_folder)