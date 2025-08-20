import cv2
import numpy as np
import configparser
import os

def read_image_path_from_config():
    # Read the image path from the configuration file
    config = configparser.ConfigParser()
    config.read('config.ini')  # Adjust the path if needed
    return config.get('ImageSettings', 'image_path', fallback=None)


def generate_contact_size_details(num_contacts, avg_radius):
    # Generate a list of contact size details for each pin
    contact_size_details = [(pin, round(avg_radius, 2)) for pin in range(1, num_contacts + 1)]
    return contact_size_details


def extract_gauge_circle_and_avg_radius(image_path):
    try:
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if image is None:
            print(f"Failed to load image at path: {image_path}")
            return None

        # Convert the image to grayscale for processing
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve circle detection
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)

        # Use adaptive thresholding to create a binary image
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use Hough Circle Transform to detect circles in the binary image
        circles = cv2.HoughCircles(
            binary_image,
            cv2.HOUGH_GRADIENT,
            dp=1,  # Inverse ratio of the accumulator resolution to the image resolution
            minDist=30,  # Minimum distance between detected centers
            param1=100,  # Higher threshold for the Canny edge detector
            param2=20,  # Accumulator threshold for the circle centers
            minRadius=5,  # Minimum circle radius to be detected
            maxRadius=50  # Maximum circle radius to be detected
        )

        # List to store circle positions, radii, and diameters
        circle_info = []

        # Check if any circles were detected
        if circles is not None:
            # Round the circle parameters to integers
            circles = np.uint16(np.around(circles))
            total_radius = 0  # Initialize total radius for average calculation

            # Loop through each detected circle
            for i in circles[0, :]:
                # Draw the detected circle on the image for visualization
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Store circle information (center and radius)
                circle_info.append({'center': (i[0], i[1]), 'radius': i[2]})
                total_radius += i[2]  # Accumulate the radius for average calculation

            total_circles = len(circle_info)  # Count the total number of detected circles
            # Calculate the average radius, ensuring no division by zero
            avg_radius = total_radius / total_circles if total_circles > 0 else 0

            # Display the image with detected circles
            cv2.imshow('Circles Detected', image)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()  # Close the image window

            return circle_info, total_circles, avg_radius  # Return detected circle info
    except cv2.error as e:
        print(f"OpenCV error: {e}")  # Handle OpenCV specific errors
    except Exception as e:
        print(f"Error during processing: {e}")  # Handle general errors
    return None  # Return None if an error occurred


# Example usage
image_path = read_image_path_from_config()  # Get the image path from the config
if image_path and os.path.isfile(image_path):  # Check if the image path is valid
    result = extract_gauge_circle_and_avg_radius(image_path)  # Extract circle information

    if result:
        circle_info, total_circles, avg_radius = result  # Unpack the result
        print("Number of Contacts:", total_circles)  # Print the number of detected circles
        #print("Gauge Size of Contacts:", round(avg_radius))  # Print the average radius
        #contact_size_details = generate_contact_size_details(total_circles,
        #                                                     round(avg_radius))  # Generate contact size details
        #print("Pin Number Contact Size Details:", contact_size_details)  # Print contact size details
    else:
        print("Extraction failed.")  # Handle extraction failure
else:
    print("Image path not found in the configuration or the file does not exist.")  # Handle invalid image path