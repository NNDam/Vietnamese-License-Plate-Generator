# Vietnamese-License-Plate-Generator
Vietnamese License Plate Generator for OCR part

# Example

Rectangle type

<p align="center">
  <img src="synthesis_sample/synthesis_3.jpg" width="200" title="hover text">
  <img src="synthesis_sample/synthesis_4.jpg" width="200" title="hover text">
  <img src="synthesis_sample/synthesis_labeled_1.jpg" width="200" title="hover text">
  <img src="synthesis_sample/synthesis_labeled_3.jpg" width="200" title="hover text">
</p>

Square type

<p align="center">
  <img src="synthesis_sample/synthesis_1.jpg" width="200" title="hover text">
  <img src="synthesis_sample/synthesis_2.jpg" width="200" title="hover text">
  <img src="synthesis_sample/synthesis_labeled_2.jpg" width="200" title="hover text">
  <img src="synthesis_sample/synthesis_labeled_4.jpg" width="200" title="hover text">
</p>

# Simple use
```
pip install -r requirements.txt
python synthesis_plate.py --numb 1000 --output_dir output
```
# Modify config
* Available characters, numbers, template in top of **synthesis_plate.py** file 
* Current generate only labels in YOLO format
