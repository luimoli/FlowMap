from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.faker import Faker
from pyecharts.globals import ChartType
import pandas as pd

# #导入excel表举例
# df=pd.read_excel('流动人口.xlsx')     
# df.head()

def get_html(df):
    geo_sight_coord={df.iloc[i]['order_id']:[df.iloc[i]['s_lng'],df.iloc[i]['s_lat']] for i in range(len(df))}
    data_pair=[(df.iloc[i]['order_id'],df.iloc[i]['d_time']) for i in range(len(df))]
    geo=Geo(init_opts=opts.InitOpts(theme='dark'))
    geo.add_schema(maptype='海口')
    for key,value in geo_sight_coord.items():
        geo.add_coordinate(key,value[0],value[1]) 
    geo.add('',data_pair,symbol_size=10,itemstyle_opts=opts.ItemStyleOpts(color="blue"))
    # 设置样式
    geo.set_series_opts(label_opts=opts.LabelOpts(is_show=False),type='scatter')
    #  is_piecewise 是否自定义分段， 变为true 才能生效
    geo.set_global_opts(visualmap_opts=opts.VisualMapOpts(),title_opts=opts.TitleOpts(title="haikou"))
    geo.render("test.html")


def get_gmplot(data):
    import gmplot
    sdata = data.sort_values('d_time_n')  #突然发现数据本身的排序并非是严格按照时间，于是还要先进行一个排序
    gmap=gmplot.GoogleMapPlotter(sdata.s_lat[0],sdata.s_lng[0],11)
    # data1=data.loc[data['ID']==1]
    gmap.plot(sdata.s_lat,sdata.s_lng)
    gmap.draw('./user001_map.html')
