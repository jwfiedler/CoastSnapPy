#This code will do the following:
#1. Get the image from this url: https://ccciaweb.ucsd.edu/imperial-beach/siocpg/CoastSnap/torrey_coastsnap_latest.jpg
#2. Run object detection on the image
from roboflow import Roboflow
rf = Roboflow(api_key="gn81NRekgIJKxcckNbx0")
project = rf.workspace("coastsnapsd").project("torrey_registration")
model = project.version(21).model

url = "https://ccciaweb.ucsd.edu/imperial-beach/siocpg/CoastSnap/torrey_coastsnap_latest.jpg"
# infer on a local image
#print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
print(model.predict(url, hosted=True, confidence=40, overlap=30).json())


