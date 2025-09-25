import qrcode
from PIL import Image, ImageDraw, ImageFont
import requests
import io

def generate_simple_qr_code(data_url: str, output_file: str = "paper_simple_qr_code.png"):
    """
    Generates a simple, square QR code for a given URL.

    Args:
        data_url (str): The URL the QR code should point to.
        output_file (str): The filename to save the QR code image.
    """
    print(f"Generating simple QR code for URL: {data_url}")
    # Create a QRCode object
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    # Add the URL data to the QR code
    qr.add_data(data_url)
    qr.make(fit=True)

    # Create an image from the QR Code instance and save it directly
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(output_file)
    
    print(f"-> Saved simple QR code as {output_file}\n")

def generate_qr_code_with_text(doi: str, paper_url: str, output_file: str = "paper_qr_code.png"):
    """Generates a QR code for a paper and adds its DOI as text below."""
    print(f"Generating QR code for paper: {doi}")
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(paper_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

    qr_width, qr_height = img.size
    img_with_text = Image.new('RGB', (qr_width, qr_height + 50), color='white')
    img_with_text.paste(img, (0, 0))

    draw = ImageDraw.Draw(img_with_text)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    text = f"DOI: {doi}"
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)

    text_x = (qr_width - text_width) // 2
    text_y = qr_height + ((50 - text_height) // 2)
    draw.text((text_x, text_y), text, fill="black", font=font)

    img_with_text.save(output_file)
    print(f"-> Saved paper QR code as {output_file}\n")

def generate_qr_code_with_logo(data_url: str, logo_url: str, output_file: str = "github_qr_code.png"):
    """Generates a QR code with a logo embedded in the center."""
    print(f"Generating QR code for repository: {data_url}")
    try:
        response = requests.get(logo_url)
        response.raise_for_status()
        logo = Image.open(io.BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        print(f"Error downloading logo: {e}")
        return

    # Create QR code with high error correction
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(data_url)
    qr.make(fit=True)
    
    # Create the QR code image in RGB mode.
    qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

    # Prepare the logo
    qr_width, qr_height = qr_img.size
    logo_max_size = qr_height // 4
    logo.thumbnail((logo_max_size, logo_max_size))
    
    # --- THE ROBUST FIX ---
    # 1. Ensure the logo is in RGBA format to guarantee it has an alpha channel
    logo = logo.convert("RGBA")
    
    # 2. Calculate the position for the logo in the center
    logo_pos = ((qr_width - logo.width) // 2, (qr_height - logo.height) // 2)

    # 3. Create a new, blank RGBA layer to paste the logo onto
    transparent_layer = Image.new("RGBA", qr_img.size, (0, 0, 0, 0))
    transparent_layer.paste(logo, logo_pos)

    # 4. Composite the QR code and the transparent layer with the logo
    final_img = Image.alpha_composite(qr_img.convert("RGBA"), transparent_layer)
    # --- END FIX ---

    # Save the final composite image
    final_img.save(output_file)
    print(f"-> Saved repository QR code as {output_file}")


# 1. Generate the simple, square QR code for the paper
paper_url = "https://link.springer.com/chapter/10.1007/978-3-032-05409-8_15"
generate_simple_qr_code(paper_url)

# 2. Generate the QR code for the paper
doi = "10.1007/978-3-032-05409-8_15"
paper_url = "https://link.springer.com/chapter/10.1007/978-3-032-05409-8_15"
generate_qr_code_with_text(doi, paper_url)

# 3. Generate the QR code for the GitHub repository
github_repo_url = "https://github.com/mrgransky/ImACCESS"
github_logo_url_png = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
generate_qr_code_with_logo(github_repo_url, github_logo_url_png)