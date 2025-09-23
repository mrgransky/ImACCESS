import qrcode
from PIL import Image, ImageDraw, ImageFont

def generate_qr_code(doi: str, paper_url: str, output_file: str = "HistoryCLIP_paper_qr_code.png"):
    """
    Generates a QR code for a paper using its DOI and URL.

    Args:
        doi (str): The DOI of the paper.
        paper_url (str): The URL of the paper.
        output_file (str): The filename to save the QR code image.
    """
    # Create a QRCode object
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    # Add the paper URL to the QR code
    qr.add_data(paper_url)
    qr.make(fit=True)

    # Create an image from the QR Code instance
    img = qr.make_image(fill_color="black", back_color="white")

    # --- FIX IS HERE ---
    # Convert the QR code image to RGB mode to match the canvas
    img = img.convert("RGB")
    # --- END FIX ---

    # Get the dimensions of the QR code image
    qr_width, qr_height = img.size

    # Create a new image with space for the DOI text
    img_with_text = Image.new('RGB', (qr_width, qr_height + 50), color='white') # Increased height for better spacing

    # Paste the QR code image onto the new image
    img_with_text.paste(img, (0, 0))

    # Use a font and draw the DOI text
    draw = ImageDraw.Draw(img_with_text)
    try:
        # Using a slightly larger font for clarity
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text position (centered)
    text = f"DOI: {doi}"
    
    # Use textbbox for more accurate size calculation in newer Pillow versions
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions
        text_width, text_height = draw.textsize(text, font=font)

    text_x = (qr_width - text_width) // 2
    text_y = qr_height + ((50 - text_height) // 2) # Center vertically in the added space

    # Draw the text
    draw.text((text_x, text_y), text, fill="black", font=font)

    # Save the image
    img_with_text.save(output_file)
    print(f"QR code saved as {output_file}")

# Example usage
doi = "10.1007/978-3-032-05409-8_15"
paper_url = "https://link.springer.com/chapter/10.1007/978-3-032-05409-8_15"
generate_qr_code(doi, paper_url)