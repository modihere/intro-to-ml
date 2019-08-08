import sys

import pyocr
import pyocr.builders
from PIL import Image

tools = pyocr.get_available_tools()
if len(tools)==0:
    print("OCR tool not found")
else:
    print([tools[i].get_name() for i in range(len(tools))])

langs = tools[0].get_available_languages()
print ("Available languages", langs)
images = 'test2.jpg'

text = tools[0].image_to_string(Image.open(images), lang=langs[0], builder=pyocr.builders.TextBuilder())

word_box = tools[0].image_to_string(Image.open(images), lang=langs[0], builder=pyocr.builders.WordBoxBuilder())

lines_word_boxes = tools[0].image_to_string(Image.open(images), lang=langs[0], builder=pyocr.builders.LineBoxBuilder())

digits = tools[0].image_to_string(Image.open(images), lang=langs[0], builder=pyocr.builders.DigitBuilder())

print(text)

if tools[0].can_detect_orientation():
    try:
        orientation = tools[0].detect_orientation(Image.open(images), lang=langs[0])
    except pyocr.PyocrException as exc:
        print(exc)
    print("Orientation: {}".format(orientation))


