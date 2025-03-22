import openxlab
from openxlab.dataset import get
# openxlab.login(ak="jqzjv0284kloqpnaxrma", sk="gmq69dxb5ndbrpozqxqeyb2l8jw8kpeqvaaglno7") # 进行登录，输入对应的AK/SK，可在个人中心添加AK/SK
openxlab.login(ak="exxj1kvmq82gd4znk8bm", sk="dq9pnvp1jndbekqyan1o59mkbz0wx4re8azlymg3")
get(dataset_repo='OpenDataLab/CityScapes', target_path='/home1/gaoheng/gh_workspace/Mask2Anomaly/datasets') # 数据集下载

# from openxlab.dataset import download
# download(dataset_repo='OpenDataLab/CityScapes',source_path='/README.md', target_path='/home1/gaoheng/gh_workspace/Mask2Anomaly/datasets') #数据集文件下载