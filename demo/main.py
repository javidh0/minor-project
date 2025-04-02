import cv2
import numpy as np
from scipy.signal import butter
from scipy import signal

import sys

from roi import getROI

import requests, base64

import matplotlib.pyplot as plt


headers = {"ngrok-skip-browser-warning": "true"}

# GET request to "/" endpoint
def get_home():
    try:
        response = requests.get(f"{base_url}/", headers=headers)
        print("GET Response:", response.text)
    except Exception as e:
        print("GET Error:", e)

# POST request to "/echo" endpoint
def post_echo(message):
    try:
        data = {"message": message}
        response = requests.post(f"{base_url}/echo", json=data, headers=headers)
        print("--------")
        print("POST Response:", response.json())
    except Exception as e:
        print("POST Error:", e)

def post_get_bpm(image_path):
    try:
        # Read and encode the image using base64
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        # Send the encoded image to the /getbpm endpoint
        data = {"image": encoded_image}
        response = requests.post(f"{base_url}/getbpm", json=data, headers=headers)

        print("--------")
        return response.json()
    except Exception as e:
        print("POST Error:", e)

def __ButterBandpass(lowcut, highcut, order):
    nyq = 0.5 * 30
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def __butterBandpassFilter(data, lowcut, highcut, order=2):
    b, a = __ButterBandpass(lowcut, highcut, order)
    y = signal.filtfilt(b, a, data)
    return y

def haveROI(r, g, b, y):
    sum_ = np.sum(r)+np.sum(g)+np.sum(b)+np.sum(y)
    if sum_ <= 2500:
        return False
    return True

def compute_first_derivative_rgb(image):
    b, g, r = cv2.split(image)
    
    def compute_gradient(channel):
        """Computes gradient magnitude using Sobel operator."""
        grad_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in X direction
        grad_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in Y direction
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)  # Gradient magnitude
        return np.mean(np.abs(grad_magnitude))  # Return mean absolute gradient value

    # Compute gradient magnitude for each channel
    r_grad = compute_gradient(r)
    g_grad = compute_gradient(g)
    b_grad = compute_gradient(b)
    
    return r_grad, g_grad, b_grad

import cv2
import numpy as np

def compute_second_derivative_rgb(image):
    """
    Computes the overall second derivative of each channel (R, G, B) of a 64x64x3 image.
    
    Parameters:
        image (numpy.ndarray): Input image of shape (64, 64, 3).
        
    Returns:
        tuple: A tuple containing three values (red, green, blue) representing the second derivatives.
    """
    # Split image into R, G, B channels
    b, g, r = cv2.split(image)
    
    # Compute Laplacian (second derivative) for each channel
    laplacian_r = cv2.Laplacian(r, cv2.CV_64F)
    laplacian_g = cv2.Laplacian(g, cv2.CV_64F)
    laplacian_b = cv2.Laplacian(b, cv2.CV_64F)
    
    # Aggregate each channel's Laplacian values (e.g., mean absolute value)
    r_value = np.mean(np.abs(laplacian_r))
    g_value = np.mean(np.abs(laplacian_g))
    b_value = np.mean(np.abs(laplacian_b))
    
    return r_value, g_value, b_value


def run(cam):
    cap = cv2.VideoCapture(cam)
    
    if not cap.isOpened():
        print("Camera not found")
        return
    
    print("recoding..")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    b = []
    g = []
    r = []
    y = []

    cnt = 0

    for _ in range(150):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(512 * frame.shape[1] / frame.shape[0]), 512))

        frame, temp = getROI(frame)
        cnt += temp

        b.append(np.mean(frame[:, :, 0]))
        g.append(np.mean(frame[:, :, 1]))
        r.append(np.mean(frame[:, :, 2]))

        ycbcr=cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)                      
        y.append(np.mean(ycbcr[:, :, 0]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    r = np.array(r[-128:])
    g = np.array(g[-128:])
    b = np.array(b[-128:])
    y = np.array(y[-128:])

    if cnt <= 120:
        print("Insufficient Data") 
        return
        
    
    r /= np.mean(r)*100
    g /= np.mean(g)*100
    b /= np.mean(b)*100
    y_norm = y/np.mean(y)*100

    r_ = __butterBandpassFilter(r, 0.7, 5)
    g_ = __butterBandpassFilter(g, 0.7, 5)
    b_ = __butterBandpassFilter(b, 0.7, 5)
    y_norm = __butterBandpassFilter(y, 0.7, 5)

    __X = 3*r_ - 2*g_
    __Y = 1.5*r_ + g_ - 1.5*b_
    __Y_lum = y

    feature_1 = []
    feature_2 = []
    feature_3 = []

    length = 128

    for i in range(length//2):
        feature_1.append(__X[i : i+(length//2)])
        feature_2.append(__Y[i : i+(length//2)])
        feature_3.append(__Y_lum[i : i+(length//2)])

    def norm(image):
        return cv2.normalize(np.array(image), None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)*255
    
    
    tempImage = cv2.merge((norm(feature_1), norm(feature_2), norm(feature_3)))
    cv2.imwrite('CHROM-Y.png', tempImage)
    
    cap.release()
    cv2.destroyAllWindows()

    res = post_get_bpm("CHROM-Y.png")
    print(res)
    

if __name__ == '__main__':
    
    global base_url
    print(sys.argv)
    base_url = sys.argv[-2]


    run(sys.argv[-1])
        
# import cv2

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# print("Press 'q' to exit.")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame. Exiting.")
#         break

#     # Resize while maintaining aspect ratio
#     height, width = frame.shape[:2]
#     aspect_ratio = width / height
#     new_width = int(512 * aspect_ratio)
#     resized_frame = cv2.resize(frame, (new_width, 512))

#     cv2.imshow('Video Capture', resized_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
