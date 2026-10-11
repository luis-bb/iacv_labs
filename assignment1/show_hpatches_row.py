import cv2
import matplotlib.pyplot as plt

# Update these paths to point to one of your sequence folders
ref_path = "hpatches-release/i_ajuntament/ref.png"
target_path = "hpatches-release/i_ajuntament/e1.png"

patch_size = 65
row_index = 5  # Change this number to look at different patches (0, 1, 2, etc.)

# 1. Load the vertical strip images
ref_strip = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
target_strip = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

# 2. Slice out the specific 65x65 patch for that row index
ref_patch = ref_strip[row_index * patch_size : (row_index + 1) * patch_size, :]
target_patch = target_strip[row_index * patch_size : (row_index + 1) * patch_size, :]

# 3. Plot them side-by-side to see the correspondence
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

axes[0].imshow(ref_patch, cmap='gray')
axes[0].set_title(f"ref.png (Row {row_index})")
axes[0].axis('off')

axes[1].imshow(target_patch, cmap='gray')
axes[1].set_title(f"e1.png (Row {row_index})")
axes[1].axis('off')
plt.show()
