from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import onnxruntime 

app = Flask(__name__)

# Load the ONNX model
onnx_model_path = 'arabic_char_ocr.onnx'
onnx_session = onnxruntime.InferenceSession(onnx_model_path)

# Mapping of numeric labels to Arabic characters and numbers
char_to_num = {
    'ا': 0, 'ب': 1, 'ت': 2, 'ث': 3, 'ج': 4, 'ح': 5, 'خ': 6, 'د': 7, 'ذ': 8, 'ر': 9,
    'ز': 10, 'س': 11, 'ش': 12, 'ص': 13, 'ض': 14, 'ط': 15, 'ظ': 16, 'ع': 17, 'غ': 18,
    'ف': 19, 'ق': 20, 'ك': 21, 'ل': 22, 'لا': 23, 'م': 24, 'ن': 25, 'ه': 26, 'و': 27,
    'ي': 28, '٠': 29, '١': 30, '٢': 31, '٣': 32, '٤': 33, '٥': 34, '٦': 35, '٧': 36,
    '٨': 37, '٩': 38
}


# Reverse mapping to get character from numeric label
num_to_char = {v: k for k, v in char_to_num.items()}

# Function to preprocess image
def preprocess_image(image_b64):
    # Decode base64 and convert to numpy array
    image_data = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode array to OpenCV image format
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Load as RGB/BGR depending on input
    
    # Determine if image is RGB or grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Convert RGB to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        # Already grayscale
        pass
    else:
        raise ValueError("Unsupported image format. Expected RGB or grayscale image.")
    
    img = cv2.resize(img, (32, 32))  # Resize to match model input shape
    img = img / 255.0     # Normalize
    img = np.expand_dims(img, axis=0)    # Add batch dimension (1, 32, 32)
    img = np.expand_dims(img, axis=-1)    # Add channel dimension for grayscale (1, 32, 32, 1)
    return img

# Route for predicting image
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image found in request'}), 400

    image_b64 = request.json['image']
    img = preprocess_image(image_b64)


    # Predict using ONNX model
    onnx_input = {onnx_session.get_inputs()[0].name: img.astype(np.float32)}
    onnx_output = onnx_session.run(None, onnx_input)
    predictions = onnx_output[0]


    # Get predicted class
    predicted_class = np.argmax(predictions)

    # Map predicted class to character
    if predicted_class in num_to_char:
        predicted_char = num_to_char[predicted_class]
    else:
        predicted_char = 'Unknown'

    return jsonify({'predicted_char': predicted_char})


if __name__ == '__main__':
    app.run(debug=True)
