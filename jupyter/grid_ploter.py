'''

Date: 2023-08-18 19:48:02
LastEditTime: 2023-08-18 20:07:52

Description: 
FilePath: /openset_anomaly_detection/jupyter/grid_ploter.py
have a nice day
'''




import matplotlib.pyplot as plt




if __name__ == "__main__":
    
    # 指定y轴和x轴的范围
    # plt.ylim(-2, 10)
    # plt.xlim(-12, 12)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    # 创建一个空的曲线图
    plt.plot([])

    # 添加网格线
    plt.grid(True)

    plt.xlabel('Penalty')
    plt.ylabel('Point-wise Penalty Loss')




    # 显示图形
    # plt.show()
    plt.savefig('energy_grid.png', dpi=300,bbox_inches='tight')

