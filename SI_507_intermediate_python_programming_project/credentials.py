import json
import requests
headers = {'Authorization': 'Bearer %s' % 'iwGMY9elHMjQqzt7rjlN9nerHSEYL-7zB0wvuMaRsOsgw_ntKrpjVlzWBwsmezjnNpaz8ypTnlEIvieJnnRAKbB3WrrVL2DSa2vE6KzElWCn_pVu6lUha7luToQ3YnYx'}

import jwt.utils

import time
import math

token = jwt.encode(
    {
        "aud": "doordash",
        "iss": "3321641e-e333-401c-9040-138ab0b457cd",
        "kid": "8523467d-9b01-4c11-bc33-d01d5807a6bc",
        "exp": str(math.floor(time.time() + 60)),
        "iat": str(math.floor(time.time())),
    },
    jwt.utils.base64url_decode("ipkCI4RmRH9JSVvf12jzVL9Ckz7mhA8mhPAHC0MB_n4"),
    algorithm="HS256",
    headers={"dd-ver": "DD-JWT-V1"})