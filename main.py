import cv2
import pytesseract
import numpy as np
from sklearn.cluster import KMeans
import random

# Configure pytesseract to point to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Change this to the path of your Tesseract executable

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def check_uniform_spacing(distances, threshold=7):
    avg_distance = np.mean(distances)
    return all(abs(dist - avg_distance) <= threshold for dist in distances)

def detect_text_spacing(image_path, y_threshold=10, x_threshold=10, spacing_threshold=5):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR using Tesseract
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Extract bounding boxes and text
    n_boxes = len(data['level'])
    boxes = []
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Confidence threshold
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append((x, y, w, h))
    
    # Analyze alignment and spacing
    if boxes:
        # Cluster boxes based on their y coordinates
        y_positions = np.array([[y] for (_, y, _, _) in boxes])
        unique_y_positions = np.unique(y_positions)
        
        lines = []
        for unique_y in unique_y_positions:
            line = [box for box in boxes if abs(box[1] - unique_y) < y_threshold]
            lines.append(sorted(line, key=lambda box: box[0]))
        
        # Check spacing within lines
        for line in lines:
            if len(line) > 1:
                x_distances = [line[i+1][0] - (line[i][0] + line[i][2]) for i in range(len(line)-1)]
                if not check_uniform_spacing(x_distances, spacing_threshold):
                    print(f"Line starting at y={line[0][1]} has non-uniform spacing: {x_distances}")
                    for i in range(len(line) - 1):
                        (x1, y1, w1, h1) = line[i]
                        (x2, y2, w2, h2) = line[i + 1]
                        cv2.line(image, (x1 + w1, y1 + h1 // 2), (x2+w2, y2 + h2 // 2), (0, 0, 255), 2)  # Red line for non-uniform spacing
                else:
                    for i in range(len(line) - 1):
                        (x1, y1, w1, h1) = line[i]
                        (x2, y2, w2, h2) = line[i + 1]
                        cv2.line(image, (x1 + w1, y1 + h1 // 2), (x2, y2 + h2 // 2), (0, 255, 0), 2)  # Green line for uniform spacing
        
        # Check spacing between lines
        if len(lines) > 1:
            line_distances = [lines[i+1][0][1] - (lines[i][0][1] + lines[i][0][3]) for i in range(len(lines)-1)]
            if not check_uniform_spacing(line_distances, spacing_threshold):
                print(f"Lines have non-uniform spacing between them: {line_distances}")
                for i in range(len(lines) - 1):
                    (_, y1, _, h1) = lines[i][0]
                    (_, y2, _, h2) = lines[i + 1][0]
                    cv2.line(image, (0, y1 + h1), (image.shape[1], y1 + h1), (255, 0, 0), 2)  # Blue line for non-uniform spacing
            else:
                for i in range(len(lines) - 1):
                    (_, y1, _, h1) = lines[i][0]
                    (_, y2, _, h2) = lines[i + 1][0]
                    cv2.line(image, (0, y1 + h1), (image.shape[1], y1 + h1), (0, 255, 0), 2)  # Green line for uniform spacing
    
    # Show the output image with detected text
    cv2.imshow('Text Spacing Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_text_alignment(image_path, y_threshold=10, x_threshold=10):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform OCR using Tesseract
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Extract bounding boxes and text
    n_boxes = len(data['level'])
    boxes = []
    for i in range(n_boxes):
        if int(data['conf'][i]) > 60:  # Confidence threshold
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append((x, y, w, h))
    
    # Analyze alignment using clustering
    if boxes:
        # Cluster boxes based on their y coordinates
        y_positions = np.array([[y] for (_, y, _, _) in boxes])
        kmeans = KMeans(n_clusters=min(len(boxes), 10)).fit(y_positions)  # Adjust number of clusters as needed
        
        clustered_boxes = {}
        for idx, label in enumerate(kmeans.labels_):
            if label not in clustered_boxes:
                clustered_boxes[label] = []
            clustered_boxes[label].append(boxes[idx])
        
        # Draw boxes with different colors based on x alignment within each cluster
        for label, cluster in clustered_boxes.items():
            # Sort cluster by x position
            cluster = sorted(cluster, key=lambda box: box[0])
            base_color = random_color()
            last_x = cluster[0][0]
            
            for (x, y, w, h) in cluster:
                if abs(x - last_x) > x_threshold:
                    base_color = random_color()  # Change color if x alignment is different
                last_x = x
                cv2.rectangle(image, (x, y), (x + w, y + h), base_color, 2)
    
    # Show the output image with detected text
    cv2.imshow('Text Alignment Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Example usage
image_path = '2.2.png'
detect_text_spacing(image_path)
