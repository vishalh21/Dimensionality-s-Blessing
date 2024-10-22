import os
import shutil

# Saving Clusterred image in the folder used for the Sample Dataset
def save_clustered_images(image_paths, labels, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)
    
    unique_labels = set(labels)
    for label in unique_labels:
        os.makedirs(os.path.join(output_dir, f"cluster_{label}"), exist_ok=True)

    for img_path, label in zip(image_paths, labels):
        img_name = os.path.basename(img_path)  # Get the image name from the path
        dest = os.path.join(output_dir, f"cluster_{label}", img_name)  # Destination path
        shutil.copy(img_path, dest)  # Copy the image to the destination

    print(f"Images have been saved to {output_dir} in respective cluster folders.")

