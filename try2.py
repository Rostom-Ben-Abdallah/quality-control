import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np


# Read images
image_path = "/content/images/testfade80.jpg"
reference_image_path = "/content/images/test2.jpeg"

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
reference_img = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)

if img is None:
    raise ValueError(f"Image not found or unable to read: {image_path}")

if reference_img is None:
    raise ValueError(f"Reference image not found or unable to read: {reference_image_path}")

# If the images have an alpha channel, remove it
if img.shape[2] == 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

if reference_img.shape[2] == 4:
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGRA2BGR)

# Instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on the image
try:
    text_ = reader.readtext(img)
except ValueError as e:
    print(f"Error reading text from image: {e}")
    raise

# Predefined expected positions (example: using the positions obtained from the previous run)
expected_positions = {
    "Anmelden": [(184, 142), (443, 142), (443, 209), (184, 209)],
    "Telefon 1": [(493, 145), (721, 145), (721, 209), (493, 209)],
    "Telefon 2": [(757, 137), (995, 137), (995, 203), (757, 203)],
    "Neustart": [(1034, 130), (1273, 130), (1273, 198), (1034, 198)],
    "Power": [(1454, 126), (1626, 126), (1626, 182), (1454, 182)],
    "Fiber": [(1729, 121), (1871, 121), (1871, 181), (1729, 181)],
    "Reset": [(1278, 396), (1432, 396), (1432, 450), (1278, 450)],
    "3x1.0Gb/s": [(451, 675), (636, 675), (636, 724), (451, 724)],
    "2.5Gb/s": [(1107, 667), (1251, 667), (1251, 711), (1107, 711)],
    "LAN1": [(248, 994), (392, 994), (392, 1050), (248, 1050)],
    "LAN2": [(492, 984), (649, 984), (649, 1047), (492, 1047)],
    "LAN3": [(746, 988), (905, 988), (905, 1040), (746, 1040)],
    "Link/LAN4": [(1029, 977), (1327, 977), (1327, 1037), (1029, 1037)],
    "USB": [(1387, 975), (1509, 975), (1509, 1035), (1387, 1035)],
    "DSL": [(1714, 976), (1826, 976), (1826, 1032), (1714, 1032)]
}

# Adjustable tolerance values
color_tolerance = 40
position_tolerance = 10

def compare_boxes(reference_img, test_img, bbox):
    x1, y1 = map(int, bbox[0])
    x2, y2 = map(int, bbox[2])
    reference_roi = reference_img[y1:y2, x1:x2]
    test_roi = test_img[y1:y2, x1:x2]
    difference = cv2.absdiff(reference_roi, test_roi)
    gray_difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, thresholded_difference = cv2.threshold(gray_difference, 55, 255, cv2.THRESH_BINARY)
    non_zero_count = np.count_nonzero(thresholded_difference)
    return non_zero_count == 0

def verify_position(actual_bbox, expected_bbox, tolerance=position_tolerance):
    return all(
        np.linalg.norm(np.array(actual) - np.array(expected)) < tolerance
        for actual, expected in zip(actual_bbox, expected_bbox)
    )

piece_defective = False
detected_positions = set()
threshold = 0.25

for t_, t in enumerate(text_):
    bbox, text, score = t

    if score > threshold:
        bbox = [tuple(map(int, point)) for point in bbox]  # Ensure the bbox coordinates are tuples of integers

        # Find the closest matching expected position
        best_match = None
        best_distance = float('inf')

        for key, expected_bbox in expected_positions.items():
            distance = np.mean([np.linalg.norm(np.array(actual) - np.array(expected)) for actual, expected in zip(bbox, expected_bbox)])
            if distance < best_distance:
                best_distance = distance
                best_match = key

        if best_match and best_distance < position_tolerance:  # Tolerance for matching position
            detected_positions.add(best_match)
            is_position_valid = verify_position(bbox, expected_positions[best_match])
            is_color_valid = compare_boxes(reference_img, img, bbox)
            print(f"Detected text at position matched with: {best_match}")
            print(f"  Color Valid: {is_color_valid}")
            print(f"  Position Valid: {is_position_valid}")
            print(f"  Detected BBox: {bbox}")
            print(f"  Expected BBox: {expected_positions[best_match]}")
        else:
            is_position_valid = False
            is_color_valid = False
            print(f"Detected text at position not matched with any expected position")
            print(f"  Color Valid: {is_color_valid}")
            print(f"  Position Valid: Not Found in Expected Positions")
            print(f"  Detected BBox: {bbox}")

        # Mark piece as defective if either condition is not valid
        if not is_color_valid or not is_position_valid:
            piece_defective = True

        color = (0, 255, 0) if is_color_valid and is_position_valid else (0, 0, 255)
        cv2.rectangle(img, bbox[0], bbox[2], color, 5)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

# Check for missing texts
missing_positions = set(expected_positions.keys()) - detected_positions
if missing_positions:
    piece_defective = True
    print(f"Missing positions: {missing_positions}")

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

if piece_defective:
    print("The piece is defective.")
else:
    print("The piece is not defective.")
