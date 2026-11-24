import os

from dotenv import load_dotenv
from google.auth import credentials
from httplib2.auth import token

load_dotenv()  # Load environment variables from .env file

from langchain_community.document_loaders import GoogleDriveLoader

credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")

loader = GoogleDriveLoader(
    folder_id="",
    credentials_path=""
    token_path=""
)


