import pandas as pd
import requests
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel  # 用于校验前端传入的坐标格式
from fastapi.middleware.cors import CORSMiddleware  # 解决跨域

# 初始化 FastAPI 应用
app = FastAPI(title="地图坐标处理服务")

# 解决跨域：允许前端地址访问（根据实际前端地址修改）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 前端运行地址（如 Vue/React 项目）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 百度地图 API 配置（复用你的密钥和参数）
AK_LIST = ["R1rtYPRvktVI5HWc1os5rMnWIwcGoKJ5"]  # 替换为实际AK
TRANSPORT_MODE = "transit"
SUBWAY_PRIORITY = 3  # 不坐地铁
COORD_TYPE = "wgs84"

# 定义前端传入的坐标格式（Pydantic 模型，自动校验）
class Coordinates(BaseModel):
    o_lat: float  # 起点纬度
    o_lon: float  # 起点经度
    d_lat: float  # 终点纬度
    d_lon: float  # 终点经度

def process_route(o_lat, o_lon, d_lat, d_lon, ak_index=0):
    """处理单条OD坐标，调用百度API获取路线（复用你的核心逻辑）"""
    try:
        current_ak = AK_LIST[ak_index]
        # 构造百度地图API请求URL
        api_url = (
            f"https://api.map.baidu.com/direction/v2/{TRANSPORT_MODE}?"
            f"origin={o_lat},{o_lon}&"
            f"destination={d_lat},{d_lon}&"
            f"coord_type={COORD_TYPE}&"
            f"ak={current_ak}&"
            f"tactics_incity={SUBWAY_PRIORITY}"
        )
        
        # 发送请求（添加适当延迟避免限流）
        response = requests.get(api_url, timeout=10)
        result = response.json()
        
        # 处理响应（复用你的逻辑）
        if result['status'] == 0:
            routes = result['result']['routes']
            if not routes:
                return {"status": "no_route", "message": "无可用路线", "data": None}
            return {
                "status": "success",
                "message": "处理成功",
                "data": routes[0]  # 返回第一条路线信息
            }
        # 密钥失效，尝试下一个AK
        elif result['status'] in [302, 401] and ak_index + 1 < len(AK_LIST):
            return process_route(o_lat, o_lon, d_lat, d_lon, ak_index + 1)
        else:
            return {
                "status": "api_error",
                "message": f"百度API错误：{result.get('message', '未知错误')}",
                "data": None
            }
    except Exception as e:
        return {"status": "error", "message": str(e), "data": None}

# 定义接收坐标的接口（POST 方法，前端通过此接口传坐标）
@app.post("/process-coords")
def process_coords(coords: Coordinates):
    """接收前端传来的起点和终点坐标，返回路线处理结果"""
    try:
        # 调用路线处理函数
        result = process_route(
            o_lat=coords.o_lat,
            o_lon=coords.o_lon,
            d_lat=coords.d_lat,
            d_lon=coords.d_lon
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"后端处理失败：{str(e)}")

# 启动服务（命令行执行：uvicorn main:app --reload --port 8000）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)