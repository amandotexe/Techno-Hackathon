<p align="center"><img src="carinsurance.png" height="340px"><br></p>
<p align="center"><h1>Car Dekhliya</h1></p>

An AI tool that streamlines your car insurance process by detecting damages with the most accuracy.

<h2>Table of Contents</h2>
- <u>Background</u>
- <u>Installation</u>
- <u>Usage</u>
- <u>Contributing</u>
- <u>License</u>

<h2>Background</h2>
Car damages can be a major safety hazard if not detected and repaired in time. However, detecting damages in cars can be a tedious and time-consuming task for humans. This is where machine learning can be of great help. By training a machine learning model on a dataset of car images with damages, we can develop an automated system for detecting damages in cars.

<h2>Installation</h2>
To install the necessary dependencies for this project, run the following command:

```
pip install -r requirements.txt
```

<h2>Usage</h2>
To use the car damage detection model, follow these steps:

- Clone this repository
- Install the necessary dependencies
- Run the prediction_engine.py script
- Provide the path to the image you want to test
- The script will output whether the car has damages or not

```
from prediction_engine import get_yolov5
model = get_yolov5(input_image)
print(model)
```

<h2>Contributing</h2>
We welcome contributions to this project. To contribute, please fork this repository, make your changes, and submit a pull request.

<h2>License</h2>
This project is licensed under the MIT License. See the LICENSE file for details.
