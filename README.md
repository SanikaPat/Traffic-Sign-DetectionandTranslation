# 🚦 Traffic Sign Detection and Translator

## Overview

The **Traffic Sign Detection and Translator** system is an intelligent application designed to enhance road safety by **detecting traffic signs** and **translating them into multiple languages in real-time**.

By combining **computer vision** and **natural language processing (NLP)**, the system provides critical road information in the driver’s preferred language — breaking down language barriers and promoting safer travel experiences.

---

## 🔍 Features

- 🧠 **Traffic Sign Recognition** using Convolutional Neural Networks (CNNs)
- 🌍 **Multilingual Translation** of detected signs using NLP
- 📷 **Camera-based Input** for real-time detection
- 🎧 **Audio Output** in selected language
- 📱 **Cross-Platform Support** – compatible with smartphones, tablets, or vehicle navigation systems
- 🌐 **Ideal for International Travelers** and **Multilingual Regions**

---

## 🛠️ How It Works

1. **Image Input**: User uploads a traffic sign image.
2. **Prediction**: The model classifies the sign using CNN-based deep learning.
3. **Translation**: The label is translated into the selected language.
4. **Audio Playback**: The translated message is converted to speech for the user to hear.

---

## 🧩 Technologies Used

- **Python + Flask** – Web application backend
- **TensorFlow / Keras** – CNN model for traffic sign classification
- **Google Translate API / gTTS** – Language translation and text-to-speech
- **HTML, CSS, JavaScript** – Frontend interface
- **Pillow, NumPy** – Image preprocessing

---

## 🚗 Use Cases

- **International Tourists** navigating foreign roads
- **Multilingual countries** with varied sign languages
- **Driver assistance systems** in smart vehicles
- **Educational tools** for learning traffic signs

---

## 💡 Future Enhancements

- Real-time camera integration
- Offline translation and TTS support
- Augmented Reality overlay for live detection
- Integration with GPS for contextual alerts

---

## 📷 Screenshots
(images/output1.png)

(images/output2.png)



---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

To run this application:

```
flask --debug run
```
