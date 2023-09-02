import requests

# 替换为您的高德地图API密钥
api_key = 'b6f48a2c0956e57df9094d44b180b98b'

# 高德地图API的URL
url = 'https://restapi.amap.com/v3/subway/line?city=北京&key={}'.format(api_key)

# 发送请求获取地铁线路信息
response = requests.get(url)
data = response.json()

# 提取地铁线路信息
subway_lines = data['l']

# 遍历每条地铁线路
for line in subway_lines:
    line_name = line['ln']
    line_id = line['ls']

    # 根据线路ID获取详细信息
    line_detail_url = 'https://restapi.amap.com/v3/subway/line?city=北京&key={}&id={}'.format(api_key, line_id)
    line_detail_response = requests.get(line_detail_url)
    line_detail_data = line_detail_response.json()

    stations = line_detail_data['s']

    # 遍历每个站点，打印经纬度信息
    for station in stations:
        station_name = station['n']
        lon, lat = station['sl'].split(',')

        print(f"Line: {line_name} - Station: {station_name} - Latitude: {lat} - Longitude: {lon}")
