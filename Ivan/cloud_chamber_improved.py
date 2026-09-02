import cv2
import numpy as np

def process_frame(img):
    """Core pipeline: takes a raw frame, returns (annotated_output, binary_mask).
    This is the exact same Otsu + bounding box geometry pipeline, just reusable."""

    # Crop out the physical borders of the chamber (Top:Bottom, Left:Right)
    img = img[100:1820, 100:980]
    
    # Keep a copy for drawing the final output
    output_img = img.copy()

    # 2. Preprocessing & Noise Reduction
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE boosts local contrast so faint electron tracks survive Otsu's threshold.
    # Without it, electron pixels are too close in intensity to the dark background
    # and Otsu's global threshold misses them entirely.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Use Median Blur to specifically target and destroy salt-and-pepper snow.
    blurred = cv2.medianBlur(enhanced, 5)

    # 3. Segmentation (Otsu's Thresholding)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Closing: bridge small gaps in fragmented electron tracks (dilate then erode).
    # This connects broken track segments into continuous contours.
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Opening: remove small isolated noise specks (erode then dilate).
    # This cleans up salt-and-pepper remnants without breaking track continuity.
    open_kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=2)

    # Black out the reflections and ceiling
    cv2.rectangle(binary, (160, 0), (250, 1920), 0, -1) 
    cv2.rectangle(binary, (660, 0), (750, 1920), 0, -1)
    cv2.rectangle(binary, (0, 0), (2000, 75), 0, -1)

    # 4. Object Detection (Contours)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Feature Extraction & Dimensional Filtering
    detections = []
    for contour in contours:
        # Get the minimum area rotated bounding box FIRST
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), angle = rect
        
        min_dim = min(width, height)
        max_dim = max(width, height)
        
        # Calculate the areas for our Density math
        box_area = min_dim * max_dim
        actual_pixel_area = cv2.contourArea(contour)
        
        if box_area == 0:
            continue

        # --- BOUNDING BOX GEOMETRY ---
        # Aspect Ratio: how elongated is the bounding box?
        # Noise blobs are round (AR ~ 1-3), tracks are elongated (AR > 3)
        aspect_ratio = max_dim / max(min_dim, 1)
        
        # Density: how much of the bounding box is filled with white pixels?
        density = actual_pixel_area / box_area

        # --- THE NOISE FILTER ---
        # Reject very tiny contours regardless of shape
        if max_dim < 25:
            continue
            
        # Kill noise: small AND not clearly elongated blobs are background snow.
        # Only keep sub-70px contours if they have high aspect ratio (clearly a track).
        if max_dim < 70 and aspect_ratio < 4:
            continue

        # 6. Classification Logic
        # Alpha tracks: thick, solid, fill most of their box, moderately elongated
        #   -> high density AND physically wide (min_dim) AND not super-elongated
        #   -> OR extremely thick (min_dim > 60) — no electron is ever that wide,
        #      but alpha glow can inflate the box and tank density below 0.40
        # Electron tracks: thin, wispy, leave empty space in their box
        #   -> low density OR physically thin OR very high aspect ratio
        if (density > 0.40 and min_dim > 15 and aspect_ratio < 8) or (min_dim > 60 and aspect_ratio < 8):
            particle_type = "Alpha"
            box_color = (0, 0, 255) # Red for Alpha
        else:
            particle_type = "Electron"
            box_color = (0, 255, 255) # Yellow for Electron

        # Draw the rotated bounding box on the output image
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(output_img, [box], 0, box_color, 2)

        # Label the box with the particle type and density
        label = f"{particle_type} (D:{density:.2f} T:{min_dim:.0f})"
        cv2.putText(output_img, label, (int(center_x), int(center_y) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        detections.append(f"Detected: {particle_type} | Length: {max_dim:.1f} | Thickness: {min_dim:.1f} | Density: {density:.2f} | AR: {aspect_ratio:.1f}")

    return output_img, binary, detections


def process_cloud_chamber_image(image_path):
    """Process a single image and display the results."""
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    output_img, binary, detections = process_frame(img)

    for d in detections:
        print(d)

    # Display the results side-by-side
    width = int(output_img.shape[1] * 0.5)
    height = int(output_img.shape[0] * 0.5)
    dim = (width, height)

    resized_binary = cv2.resize(binary, dim, interpolation=cv2.INTER_AREA)
    resized_output = cv2.resize(output_img, dim, interpolation=cv2.INTER_AREA)

    # Save results to disk for verification
    cv2.imwrite("result_binary.jpg", resized_binary)
    cv2.imwrite("result_classified.jpg", resized_output)
    print("Saved: result_binary.jpg and result_classified.jpg")

    cv2.imshow("Binary Threshold (Otsu)", resized_binary)
    cv2.imshow("Track Classification", resized_output)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_cloud_chamber_video(video_path):
    """Process a video frame-by-frame and display live results. Press 'Q' to quit."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Delay between frames in ms (match original video speed)
    delay = max(1, int(1000 / fps)) if fps > 0 else 30

    print(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    print("Press 'Q' to quit, 'P' to pause/unpause.")

    frame_num = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            frame_num += 1
            output_img, binary, detections = process_frame(frame)

            if detections:
                print(f"\n--- Frame {frame_num}/{total_frames} ---")
                for d in detections:
                    print(d)

            # Resize for display
            width = int(output_img.shape[1] * 0.5)
            height = int(output_img.shape[0] * 0.5)
            dim = (width, height)

            resized_output = cv2.resize(output_img, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("Track Classification (Video)", resized_output)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("Stopped by user.")
            break
        elif key == ord('p') or key == ord('P'):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    cap.release()
    cv2.destroyAllWindows()


# --- UI: File picker dialog ---
import tkinter as tk
from tkinter import filedialog

print("Opening file dialog... Please select an image or video.")
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

selected_file = filedialog.askopenfilename(
    title="Select a Cloud Chamber Image or Video",
    initialdir="Raw_Dataset",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png"),
        ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
        ("All Files", "*.*")
    ]
)
root.destroy()

if selected_file:
    print(f"Processing: {selected_file}")
    # Route to the right function based on file extension
    ext = selected_file.lower().rsplit('.', 1)[-1]
    if ext in ("mp4", "avi", "mov", "mkv"):
        process_cloud_chamber_video(selected_file)
    else:
        process_cloud_chamber_image(selected_file)
else:
    print("No file was selected.")
