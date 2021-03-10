"""
Command line tool that takes an image, detecs the faces, and crops them
Saves cropped faces to local dir
Usage: python cropface.py --fname {image-filename}
"""

import cv2
import click


def predict_sentiment():
    return 'sentiment'

def detect_face(img, verbose=False):
    """
    Detect faces using the OpenCV haar cascade classifier
    Returns a list of face coordinates
    """
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    face_cnt = len(faces)
    if verbose == True:
        print(f"Detected faces: {face_cnt}")
    return faces

def box_faces(img,faces,size=' ',label=True):
    """
    Draws a box around the faces in the image file
    Use the size parameter to set a lower bound for detected face size
    No return
    """
    for (x,y,w,h) in faces:
        if type(size) == str:
            lim = w
        else:
            lim = size

        if w >= lim:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            if label == True:
                text = predict_sentiment()
                cv2.putText(img, text, (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)

def show_image(img_name, img, waitKey=0):
    """
    Shows cv2.imshow() with waitKey(0) and destroyAllWindows
    """
    cv2.imshow(img_name, img)
    cv2.waitKey(waitKey)
    cv2.destroyAllWindows()

def crop(img,face):
    """
    Takes the image and crops the faces
    Returns the cropped face
    """
    height, width = img.shape[:2]
    x, y, w, h = face
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    faceimg = img[ny:ny+nr, nx:nx+nr]
    lastimg = cv2.resize(faceimg, (256, 256))
    return lastimg

@click.command()
@click.option('--fname')
def crop_face(fname):
    img = cv2.imread(fname)

    faces = detect_face(img,verbose=True)

    # box_faces(img,faces)

    show_image('img', img)

    face_captures = []
    i=0
    for face in faces:
        lastimg = crop(img,face)
        i += 1
        filename = f'image{i}.jpg'
        face_captures.append(filename)
        cv2.imwrite(filename, lastimg)

    for image in face_captures:
    	face = cv2.imread(image)
    	show_image('face', face)

if __name__ == "__main__":
	crop_face()
