from PIL import Image
import pytesseract


# Open an image file
image_path = 'idk.png'
img = Image.open(image_path)

# Use Tesseract to do OCR on the image
extracted_text = pytesseract.image_to_string(img)

print("Extracted Text:", extracted_text)
