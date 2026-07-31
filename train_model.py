from imutils import paths
import face_recognition
import pickle
import cv2
import os

# hinh anh cua chung ta duoc luu trong thu muc dataset
print("[INFO] bat dau xu ly cac khuon mat...")
imagePaths = list(paths.list_images("dataset"))

# khoi tao danh sach cac dac trung khuon mat (encodings) va ten da biet
knownEncodings = []
knownNames = []

# lap qua cac duong dan hinh anh
for (i, imagePath) in enumerate(imagePaths):
    # trich xuat ten nguoi tu duong dan hinh anh
    print("[INFO] dang xu ly hinh anh {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # tai hinh anh dau vao va chuyen doi no tu he mau BGR (thu tu mac dinh cua OpenCV)
    # sang he mau RGB (thu tu cua dlib)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # phat hien toa do (x, y) cua cac khung bao (bounding boxes)
    # tuong ung voi tung khuon mat trong hinh anh dau vao
    boxes = face_recognition.face_locations(rgb, model="hog")

    # tinh toan vector dac trung (facial embedding) cho tung khuon mat
    encodings = face_recognition.face_encodings(rgb, boxes)

    # lap qua cac dac trung vua tim duoc
    for encoding in encodings:
        # them tung dac trung + ten vao danh sach ten va dac trung da biet
        knownEncodings.append(encoding)
        knownNames.append(name)

# luu tru (dump) cac dac trung khuon mat + ten xuong o cung
print("[INFO] dang tuan tu hoa (serializing) cac dac trung...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()