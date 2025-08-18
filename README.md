# Fire Detection Binary Classifier

This project implements a binary image classifier to detect forest fires using deep learning with TensorFlow and Keras. The workflow is organized in a Jupyter notebook (`fire_detection.ipynb`).

## Dataset
- Source: [Kaggle - Forest Fire Dataset](https://www.kaggle.com/datasets/datascientist97/forest-fire/data)
- Data is split into training and testing sets, each containing images and YOLO-format label files.

## Workflow
1. **Imports**: Loads required libraries (PyTorch, NumPy, Matplotlib, PIL, etc.).
2. **Data Loading**: Custom function loads images and labels, matches them, and preprocesses images to 128x128 RGB arrays.
3. **Data Augmentation**: Uses `ImageDataGenerator` for basic augmentation (rotation, zoom, horizontal flip).
4. **Model Architecture**: Sequential CNN with three convolutional layers, max pooling, and dense layers for binary classification.
5. **Training**: Model is trained for 20 epochs using the augmented data.
6. **Evaluation**: Prints final test accuracy and loss.
7. **Saving**: Trained model is saved as `fire_classifier.keras`.
8. **Visualization**: Random test images are shown with predicted and actual labels, and prediction probabilities.

## Usage
- Update the dataset paths in the notebook if needed.
- Run the notebook cells sequentially to train and evaluate the model.
- The saved model can be loaded for future inference.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Pillow

Install dependencies with:
```bash
pip install torch numpy matplotlib scikit-learn pillow
```

## Notes
- The notebook suppresses some warnings for cleaner output.
- The classifier predicts "Fire" or "No Fire" for each image.

## License
This project is for educational purposes. Please check the dataset license before commercial use.
