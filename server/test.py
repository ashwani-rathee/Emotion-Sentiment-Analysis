import requests
from skimage import io
import json
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

urllib.request.urlretrieve("https://i.imgur.com/MWC4ywP.png", "local-filename.jpg")
resp = requests.post("http://localhost:5000/predict",
                     files={"file":open('local-filename.jpg','rb')})

# resp = requests.post("https://brain-segment-api.herokuapp.com/predict",
#                      files={"file":open('local-filename.jpg','rb')})
# resp = requests.post("https://brain-tumor-segment-api.as.r.appspot.com/predict",
#                      files={"file":open('local-filename.jpg','rb')})

json_load= resp.json()
a_restored = np.asarray(json_load["mask"])
print(a_restored)
# io.imshow(a_restored)
# io.show()