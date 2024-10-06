import numpy as np
import datetime
from matplotlib import pyplot as plt
import netCDF4 as nc
import numpy as np

start_date = datetime.date(2000, 6, 1)
end_date = datetime.date(2024, 4, 30)
delta = datetime.timedelta(days=1)
days_accumulated = 0
week = 1
while start_date <= end_date:
  formatted_date = start_date.strftime("%Y%m%d")
  url = "/content/drive/MyDrive/precipitation/content/"
  file_path = "3B-DAY.MS.MRG.3IMERG."+str(formatted_date)+"-S000000-E235959.V07B.nc4.nc4?precipitation[0:0][552:867][1159:1449],time,lon[552:867],lat[1159:1449]"
  dataset = nc.Dataset(url+file_path)
  variable = dataset.variables['precipitation']  # Replace with the variable name
  data = variable[:]
  data = data.reshape(316,291) # used print(data.shape) to find shape
  if days_accumulated == 0:
    data_256_256 = np.zeros((256,256),dtype=np.float32)
    data_256_256 = data_256_256+data[25:25+256,18:18+256]
    days_accumulated = days_accumulated + 1
  else:
    data_256_256 = data_256_256+data[25:25+256,18:18+256]
    days_accumulated = days_accumulated + 1
  if days_accumulated == 7:
    np.save("/content/drive/MyDrive/precipitation/"+str(week),data_256_256.data)
    week = week+1
    days_accumulated = 0
  start_date += delta
