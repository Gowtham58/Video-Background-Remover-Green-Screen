# Video-Backgroud-Remover-Green-Screen
Removes your video background and adds green screen Using the briaai/RMBG-1.4 from HuggingFace 
# Methodology
Uses the cv2 library to read a video frame by frame. Then removes the background from every frame using the RMBG model from Briaai in HuggingFace

# User Interface
The UI was built using Gradio

![Image](https://github.com/Gowtham58/Video_Backgroud_Remover_Green_Screen/blob/main/asset/demo.png)

# INPUT
https://github.com/Gowtham58/Video_Backgroud_Remover_Green_Screen/assets/75661938/fb14f68c-7326-46ad-bf00-d76e0c535ca3


# OUTPUT
https://github.com/Gowtham58/Video_Backgroud_Remover_Green_Screen/assets/75661938/44f5de3c-9083-4004-a600-043ba3645af4


## TO RUN LOCALLY 
1. Download the repository or clone it locally
2. Create a python virtual environment using ```python -m venv```
3. Once Created Activate the virtual environment
4. Install the packages from the requierments.txt ```pip install -r requirements```
5. Run the application using ```python app.py```
