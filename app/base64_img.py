import base64
from io import BytesIO

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="jpeg")
    img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
    return 'data:image/jpeg;base64,' + img_str