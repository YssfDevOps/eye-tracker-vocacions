# Eye Tracker for Academic Vocations

This project implements an **eye tracker using a webcam** to analyze implicit preferences of high school students. The goal is to compare explicit career choices with subconscious interests, providing a **more accurate vocational guidance tool**.

## Features
- **Real-time eye tracking** using `dlib` and `OpenCV`
- **Machine learning models** for analyzing gaze patterns
- **Data visualization** to compare explicit vs. implicit interests
- **User-friendly interface** for students and educators

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/YOUR_USERNAME/eye-tracker-vocacions.git
cd eye-tracker-vocacions
```

### 2. Set Up a Virtual Environment
```sh
python -m venv venv
```
#### Windows (PowerShell):
```sh
.\venv\Scripts\Activate
```
#### macOS/Linux:
```sh
source venv/bin/activate
```

### 3. Install Dependencies
Make sure you have CMake and Visual Studio Build Tools installed (**for `dlib`**). Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

## Dependencies
- `pandas` – Data processing
- `torch`, `torchvision` – Machine learning models
- `PIL` – Image processing
- `opencv-python` – Computer vision tasks
- `dlib` – Face and eye tracking
- `numpy`, `matplotlib`, `scipy` – Data analysis

## Usage
Run the main script to start eye tracking:
```sh
python main.py
```

## Folder Structure
```
📁 eye-tracker-vocacions
│── 📂 src/            # Source code files
│── 📂 models/         # ML models and training data
│── 📂 data/           # Sample datasets
│── 📜 requirements.txt # Project dependencies
│── 📜 README.md       # Project documentation
│── 📜 main.py         # Main script to run the tracker
```

## Contributing
1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Commit your changes**: `git commit -m "Add new feature"`
4. **Push to GitHub**: `git push origin feature-name`
5. **Open a Pull Request**

## License
This project is licensed under the **MIT License**.

## Contact
For questions or suggestions, reach out via GitHub Issues or email at **your-email@example.com**.

---
