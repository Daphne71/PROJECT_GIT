#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# externalIP = os.popen('curl -s ifconfig.me').readline()

# if externalIP == "211.57.136.84":  # Linux server Externel IP 84 / 작업 pc : 211.57.136.82
#     DLS002_HOST = '10.1.55.202'  # 회사 내부
# else:
#     DLS002_HOST = "211.57.136.88"  # 회사 외부

DLS002 = {
    "HOST": '10.1.55.202',
    "DB": 'AI_solution',
    "USER": 'erp_test',
    "PW": '*Dlit_Erptest#7004!'
}

