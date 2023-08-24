import sys
import cv2
import imutils
import numpy as np
import supervision as sv
from ultralytics import YOLO
from src.helper import read_s3_image, write_s3_image, stretch_and_scale

bucket_name = "license-replace"

def handler(event, context):
  bucket_name = event["bucket_name"]
  car_image_path = event["car_image_path"]

  if "logo_image_path" in event:
      logo_image_path = event["logo_image_path"]
  else:
      logo_image_path = './static/allo_logo_rounded_border_01.png'

  # read image from s3 bucket
  car_image = read_s3_image(bucket_name, car_image_path)
  car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2BGR)

  model = YOLO('./static/license-segment.pt')
  result = model(car_image)[0]
  detections = sv.Detections.from_ultralytics(result)

  mask = detections.mask.squeeze()

  int_mask = mask.astype(np.uint8)
  int_mask_copy = int_mask.copy()
  contours = cv2.findContours(int_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(contours)
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

  locations = []
  for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 5, True)
    if len(approx) == 4:
      locations.append(approx)

  if (len(locations) > 0):
    corners = locations[0]

  # print('corners')
  # print(corners)

  modified_corners = stretch_and_scale(corners)

  # print('modified_corners')
  # print(modified_corners)

  allo_logo = cv2.imread(logo_image_path)
  rounded_mask_white = cv2.imread('static/rounded_mask_white.png')

  bottom_left = [0, allo_logo.shape[0]]
  top_left = [0,0]
  bottom_right = [allo_logo.shape[1], allo_logo.shape[0]]
  top_right = [allo_logo.shape[1], 0]
  pts_src = np.array([bottom_left, top_left, top_right, bottom_right])

  h, status = cv2.findHomography(pts_src, modified_corners, cv2.RANSAC)

  plate_isolated = cv2.warpPerspective(allo_logo, h, (car_image.shape[1], car_image.shape[0]))
  rounded_mask_warped = cv2.warpPerspective(rounded_mask_white, h, (car_image.shape[1], car_image.shape[0]))

  mask_inv = cv2.cvtColor(cv2.bitwise_not(rounded_mask_warped), cv2.COLOR_BGR2GRAY)

  img_bo = cv2.bitwise_and(car_image, car_image, mask=mask_inv)
  img_bo = cv2.cvtColor(img_bo, cv2.COLOR_BGR2RGB)
  final = cv2.bitwise_or(img_bo, plate_isolated)


  result_image_string = cv2.imencode(".jpg", final)[1].tobytes()
  write_s3_image(result_image_string, bucket_name, "result.jpg")