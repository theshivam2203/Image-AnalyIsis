import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    # Convert numpy array to PIL Image
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    # Apply transformation
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)
    # Extract features
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

def calculate_color_percentages(image):
    # Convert BGR to RGB if necessary
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Threshold for white, light gray, and dark gray
    _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    _, light_gray_mask = cv2.threshold(gray, 150, 239, cv2.THRESH_BINARY)
    _, dark_gray_mask = cv2.threshold(gray, 50, 149, cv2.THRESH_BINARY)
    
    # Calculate areas
    total_pixels = image.shape[0] * image.shape[1]
    white_area = np.sum(white_mask) / total_pixels * 100
    light_gray_area = np.sum(light_gray_mask) / total_pixels * 100
    dark_gray_area = np.sum(dark_gray_mask) / total_pixels * 100
    
    return white_area, light_gray_area, dark_gray_area

def calculate_intensity(image):
    # Convert BGR to RGB if necessary
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Calculate mean intensity
    intensity = np.mean(gray)
    
    return intensity

def calculate_hue(image):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate mean hue
    hue = np.mean(hsv[:,:,0])
    
    return hue

def calculate_angles_and_curvature(image):
    # Dummy function for demonstration, replace with actual implementation
    # Here we assume we calculate angles and curvature
    angles = np.random.rand() * 180  # Replace with actual calculation
    curvature = np.random.rand() * 10  # Replace with actual calculation
    
    return angles, curvature

def compare_cells_ml(image1, image2, grid_size):
    h, w, _ = image1.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    differences = []

    for i in range(grid_size):
        row_differences = []
        for j in range(grid_size):
            cell1 = image1[i * cell_h: (i + 1) * cell_h, j * cell_w: (j + 1) * cell_w]
            cell2 = image2[i * cell_h: (i + 1) * cell_h, j * cell_w: (j + 1) * cell_w]

            # Ensure cell dimensions are within image bounds
            if cell1.shape[0] == 0 or cell1.shape[1] == 0 or cell2.shape[0] == 0 or cell2.shape[1] == 0:
                continue

            # Extract features using the CNN
            features1 = extract_features(cell1)
            features2 = extract_features(cell2)

            # Calculate similarity using cosine similarity
            similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))[0][0]

            # Calculate color percentages
            white1, light_gray1, dark_gray1 = calculate_color_percentages(cell1)
            white2, light_gray2, dark_gray2 = calculate_color_percentages(cell2)

            # Calculate intensity
            intensity1 = calculate_intensity(cell1)
            intensity2 = calculate_intensity(cell2)

            # Calculate hue
            hue1 = calculate_hue(cell1)
            hue2 = calculate_hue(cell2)

            # Calculate angles and curvature
            angles1, curvature1 = calculate_angles_and_curvature(cell1)
            angles2, curvature2 = calculate_angles_and_curvature(cell2)

            color_difference = {
                "similarity": similarity,
                "white_diff": white1 - white2,
                "light_gray_diff": light_gray1 - light_gray2,
                "dark_gray_diff": dark_gray1 - dark_gray2,
                "intensity_diff": intensity1 - intensity2,
                "hue_diff": hue1 - hue2,
                "angles_diff": angles1 - angles2,
                "curvature_diff": curvature1 - curvature2
            }
            row_differences.append(color_difference)
        differences.append(row_differences)
    return differences

def visualize_differences(image1, image2, grid_size, differences):
    combined_image = np.concatenate((image1, image2), axis=1)
    cell_h, cell_w, _ = image1.shape

    texts = []

    for i in range(grid_size):
        for j in range(grid_size):
            if i < len(differences) and j < len(differences[i]) and "similarity" in differences[i][j]:
                similarity = differences[i][j]["similarity"]
                white_diff = differences[i][j]["white_diff"]
                light_gray_diff = differences[i][j]["light_gray_diff"]
                dark_gray_diff = differences[i][j]["dark_gray_diff"]
                intensity_diff = differences[i][j]["intensity_diff"]
                hue_diff = differences[i][j]["hue_diff"]
                angles_diff = differences[i][j]["angles_diff"]
                curvature_diff = differences[i][j]["curvature_diff"]

                text1 = f"Cell {i},{j}:"
                text2 = f"  Similarity: {similarity:.3f}"
                text3 = f"  White Diff: {white_diff:.2f}%"
                text4 = f"  Light Gray Diff: {light_gray_diff:.2f}%"
                text5 = f"  Dark Gray Diff: {dark_gray_diff:.2f}%"
                text6 = f"  Intensity Diff: {intensity_diff:.2f}"
                text7 = f"  Hue Diff: {hue_diff:.2f}"
                text8 = f"  Angles Diff: {angles_diff:.2f}"
                text9 = f"  Curvature Diff: {curvature_diff:.2f}"

                text = f"{text1}\n{text2}\n{text3}\n{text4}\n{text5}\n{text6}\n{text7}\n{text8}\n{text9}\n"
                texts.append(text)

    return combined_image, texts

def main():
    st.title("Image Comparison and Analysis")
    st.write("Upload two images to compare them.")

    uploaded_file1 = st.file_uploader("Upload Image 1", type=['jpg', 'png', 'jpeg'])
    uploaded_file2 = st.file_uploader("Upload Image 2", type=['jpg', 'png', 'jpeg'])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        image1 = np.array(Image.open(uploaded_file1))
        image2 = np.array(Image.open(uploaded_file2))

        grid_size = st.slider("Select grid size for comparison", 2, 5, 3)
        differences = compare_cells_ml(image1, image2, grid_size)
        combined_image, texts = visualize_differences(image1, image2, grid_size, differences)

        st.subheader("Comparison Result:")
        st.image(combined_image, caption="Side-by-Side Comparison", use_column_width=True)

        st.subheader("Detailed Differences:")
        for text in texts:
            st.text(text)
    else:
        st.write("Please upload both images.")

if __name__ == "__main__":
    main()



