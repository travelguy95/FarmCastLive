# LINKS.txt is list of links to your desired data. This can be downloaded from 
# https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary?keywords=IMERG

with open('LINKS.txt', 'r') as file:
    text = file.read()
    lines = text.splitlines()

sum = 0
link_list = []
for line in lines:
  sum = sum + 1
  link_list.append(line)

import requests

class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'
    def __init__(self, username, password):
        super().__init__()
        self.auth = (username, password)
    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers
        url = prepared_request.url
        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)
            if (original_parsed.hostname != redirect_parsed.hostname) and \
                    redirect_parsed.hostname != self.AUTH_HOST and \
                    original_parsed.hostname != self.AUTH_HOST:
                del headers['Authorization']
        return

username = "USERNAME"
password= "PASSWORD"
session = SessionWithHeaderRedirection(username, password)

for link in range(len(link_list)):
  try:
    url = link_list[link]
    filename = url[url.rfind('/')+1:]
    response = session.get(url, stream=True)
    print(response.status_code)
    response.raise_for_status()
    with open(filename, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=1024*1024):
            fd.write(chunk)

  except requests.exceptions.HTTPError as e:
    print(e)
