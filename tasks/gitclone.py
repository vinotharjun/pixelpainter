import os
from getpass import getpass
import urllib.parse
import sys

user = str(input("enter username"))
password = getpass("enter password")
password = urllib.parse.quote(
    password)  # your password is converted into url format
repo_name = "pixelpainter"

cmd_string = 'git clone https://{0}:{1}@github.com/{0}/{2}.git'.format(
    user, password, repo_name)

os.system(cmd_string)
cmd_string, password = "", ""  # removing the password from the varia
