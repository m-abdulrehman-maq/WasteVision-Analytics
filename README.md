# Garbage Classification System

The Garbage Classification System is an automated computer vision tool designed to enhance waste sorting and environmental management. Powered by the Ultralytics YOLOv8 architecture and trained using datasets from Roboflow, this project empowers you to effortlessly detect and classify various waste materials (such as plastic, metal, glass,cardboard and organic waste) from images and visual data.

## Features

* **Custom Object Detection:** This model allows you to quickly detect and classify multiple categories of waste. No more manual sorting or tedious visual checks; you can instantly identify different waste types.
* **Accurate Confidence Scoring:** In addition to detecting objects, the model provides precise confidence scores for each bounding box to give you a comprehensive overview of prediction reliability.
* **User-Friendly Workflow:** The project seamlessly integrates into standard Python environments, providing a clean and intuitive setup using Ultralytics YOLOv8. It's easy to use, even for those who may not be deep learning experts.
* **Quick Access & Inference:** With just a few lines of code, you can run predictions on new images without the need to navigate through complex scripts or perform manual model configurations.

## Installation & Setup

To set up and run the Garbage Classification System, follow these simple steps:

1. **Clone or Download Repository:** First, clone this GitHub repository or download it as a ZIP file to your local machine.
2. **Install Dependencies:**
   * Open your terminal or Python environment.
   * Run the command to install required packages:
     ```bash
     pip install ultralytics roboflow opencv-python
     ```
3. **Using the Model:**
   * Once your dependencies are installed, ensure your trained weights file (`best.pt`) is in the project directory.
   * Use the provided Python script to test the model on any custom image.

## Running Predictions

To test the trained model on an image, use the following code snippet:

```python
from ultralytics import YOLO

# Load your trained model weights
model = YOLO('best.pt')

# Run prediction on an image
results = model.predict(source='path_to_your_image.jpg', save=True, conf=0.25)

print("Detection complete! Check the 'runs/detect/predict' folder for output results.")
```
## Feedback and Contributions
We welcome your feedback and contributions to improve the Garbage Classification System. If you encounter any issues, have suggestions for new features, or want to contribute to the project, please don't hesitate to open an issue or submit a pull request on this GitHub repository.
## License
This project is licensed under the MIT License. Feel free to modify and distribute it in accordance with the license terms.


