Title:    Real-Time Sign Language Recognition Using MobileNetV2 and Hand Tracking

Abstract:
Sign language is a crucial means of communication for people with hearing or speech impairments, but it remains less understood by the general public. To bridge this gap, we developed a system that can recognize sign language gestures in real-time and translate them into text and speech. Our approach uses MobileNetV2, a lightweight yet powerful deep learning model, and MediaPipe for hand tracking. The system is efficient, accurate, and designed to run smoothly on everyday devices.

Introduction:
Sign language translation is vital for creating inclusive communication channels for those with hearing or speech disabilities. Despite advancements in AI, a real-time, accessible, and efficient solution for translating sign language gestures remains a challenge. In this paper, we discuss our effort to develop a system that uses deep learning and hand tracking to recognize gestures and convert them into understandable outputs like text and speech.

Dataset and Preprocessing
Dataset:
We collected images of various sign language gestures and organized them into separate folders for each class. The dataset was divided into training and validation sets to ensure the model could generalize well.
Augmentation and Normalization
To make the model more robust, we augmented the dataset by applying transformations like rotation, zoom, width/height shifts, and horizontal flips. This helped simulate real-world variations in gesture positions. The images were also normalized to scale pixel values between 0 and 1, making them suitable for model input.

Model Architecture
Base Model: MobileNetV2
We used MobileNetV2 as the core model because it is lightweight and performs well on limited hardware. The pre-trained weights from ImageNet allowed us to leverage transfer learning, which sped up the training process and improved accuracy.
Adding Layers
To customize the model for our task, we added layers on top of MobileNetV2:
•	A GlobalAveragePooling2D layer to reduce dimensions.
•	Fully connected layers with 512 and 256 neurons for better feature extraction.
•	Dropout layers to reduce overfitting.
•	A final softmax layer for classifying gestures.

Training the Model
Training Process:
The training process started with the pre-trained MobileNetV2 layers frozen. This allowed us to train only the newly added layers. Once the model stabilized, we fine-tuned the last 100 layers of the base model for improved performance.
Key Parameters
•	Optimizer: Adam (learning rate of 0.0001).
•	Loss Function: Categorical crossentropy.
•	Metrics: Accuracy.

Results:
The model achieved high accuracy on the validation set and showed consistent improvement during fine-tuning.

Real-Time Testing
Hand Tracking:
We integrated MediaPipe for hand tracking, which efficiently detects hand landmarks. This helped us crop the hand region dynamically for each frame, ensuring real-time adaptability.
Gesture Classification
The cropped image was resized to the model’s input dimensions and passed through the trained model for prediction. The system used confidence scores to filter out uncertain predictions.
User Interface
The GUI was built using Tkinter, allowing users to see real-time predictions, form words by combining gestures, and hear the output using text-to-speech. The interface is simple and user-friendly, designed for easy interaction.

Challenges:
Similar Gestures
Some gestures looked similar, causing confusion during classification. To address this, we expanded the dataset and fine-tuned the model further.
Real-Time Efficiency
Ensuring smooth real-time performance was tricky, especially with limited hardware. We kept the architecture lightweight and optimized the preprocessing pipeline.

Results:
•	Training Accuracy: The model achieved over 65% accuracy on the validation set.
•	Real-Time Performance: The system processed live video input efficiently, with gesture predictions appearing almost instantly.
•	User Experience: The interface was intuitive, and users found it helpful in translating gestures into understandable text and speech.

Conclusion:
This project demonstrates that sign language recognition can be made efficient and accessible using a combination of deep learning and hand tracking. While our current model handles a fixed set of gestures, future work can focus on expanding the vocabulary and improving recognition in complex environments.
