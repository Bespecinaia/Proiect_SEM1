from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

not_keep_list = ["Card", "Total"]

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input receipt image")
ap.add_argument("-d", "--debug", type=int, default=-1,
                help = "wheter or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

orig = cv2.imread(args["image"])
image = orig.copy()
image = imutils.resize(image, width=500)
ratio = orig.shape[1] / float(image.shape[1])

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5,), 0)
edged = cv2.Canny(blurred, 75, 200)

if args["debug"] > 0:
    cv2.imshow("Input", image)
    cv2.imshow("Edget", edged)
    cv2.waitKey()

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

receiptCnt = None

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        receiptCnt = approx
        break

if receiptCnt is None:
    raise Exception(("Could not find receipt outline. "
                     "Try debugging your edge detection and contour steps."))

if args["debug"] > 0:
    output = image.copy()
    cv2.drawContours(output, [receiptCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Receipt Outline", output)
    cv2.waitKey(0)

reciept = four_point_transform(orig, receiptCnt.reshape(4, 2) * ratio)
cv2.imshow("Receipt Transform", imutils.resize(reciept, width=500))
cv2.waitKey(0)

options = "--psm 4"
text = pytesseract.image_to_string(
    cv2.cvtColor(reciept, cv2.COLOR_BGR2RGB),
    config=options
)

pricePattern = r'([0-9]+\.[0-9]+)'
#quantity_pattern = r'([0-9]+\s*[xX]\s*[0-9]+)'
# show the output of filtering out *only* the line items in the
# receipt

having_price = []
for row in text.split("\n"):
    # check to see if the price regular expression matches the current
    # row
    if re.search(pricePattern, row) is not None:
        having_price.append(row)
reciept_prices = {}

for row in having_price:
    elements = row.split()
    split_index = 0
    for i in range(len(elements)):
        if re.search(pricePattern, elements[i]) is not None:
            split_index = i
    reciept_prices[' '.join(elements[:split_index])] = float(elements[split_index].replace("$", ""))

to_delete_keys = []
for key in reciept_prices:
    for not_keep in not_keep_list:
        if not_keep in key:
            to_delete_keys.append(key)
for key in to_delete_keys:
    del reciept_prices[key]
print(reciept_prices)