import cv2 as cv
from PIL import Image

import mtcnn.detector as mtcnn
import retinaface.detector as retinaface


def show_bboxes(full_path, bboxes, landmarks):
    img_raw = cv.imread(full_path)
    num_faces = bboxes.shape[0]

    # show image
    for i in range(num_faces):
        b = bboxes[i]
        width = b[2] - b[0]
        height = b[3] - b[1]
        area = width * height
        print('width: ' + str(width))
        print('height: ' + str(height))
        print('area: ' + str(area))
        scores = bboxes[:, 4]
        text = "{:.4f}".format(scores[i])
        b = list(map(int, b))
        cv.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv.putText(img_raw, text, (cx, cy),
                   cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # landms
        landms = landmarks[i]
        cv.circle(img_raw, (landms[0], landms[5]), 1, (0, 0, 255), 4)
        cv.circle(img_raw, (landms[1], landms[6]), 1, (0, 255, 255), 4)
        cv.circle(img_raw, (landms[2], landms[7]), 1, (255, 0, 255), 4)
        cv.circle(img_raw, (landms[3], landms[8]), 1, (0, 255, 0), 4)
        cv.circle(img_raw, (landms[4], landms[9]), 1, (255, 0, 0), 4)

    # save image

    cv.imwrite('images/result.jpg', img_raw)
    cv.imshow('image', img_raw)
    cv.waitKey(0)


if __name__ == "__main__":
    full_path = 'test/Jason Behr_27968.JPG'
    img = Image.open(full_path).convert('RGB')
    bboxes, landmarks = mtcnn.detect_faces(img)
    print(bboxes)
    print(landmarks)
    show_bboxes(full_path, bboxes, landmarks)

    bboxes, landmarks = retinaface.detect_faces(img)
    print(bboxes)
    print(landmarks)
    show_bboxes(full_path, bboxes, landmarks)
