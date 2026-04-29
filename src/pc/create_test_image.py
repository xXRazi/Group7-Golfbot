from Imagesplitter import create_matrix
from dotenv import load_dotenv
import os

load_dotenv()
img_path = os.getenv("img_path")

test_path = img_path + "/4.png"

test_matrix = create_matrix(test_path)