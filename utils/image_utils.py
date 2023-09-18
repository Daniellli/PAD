


from skimage import measure
import cv2


import matplotlib.pyplot as plt
import torch


from loguru import logger

import torch.nn.functional as F 
import os 
import os.path as osp

from os.path import split,join,exists
from PIL import Image
import numpy as np

'''
description: make sure the type of img is acceptable image  
param {*} img
return {*}
'''
def check_img(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img




def convert_image_vertical(img):

    #* turn into image 
    im = Image.fromarray(img)
    flip_left_right = im.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
    # flip_top_bottom = im.transpose(Image.FLIP_TOP_BOTTOM)  # 垂直翻转
    #* turn back 
    return np.array(flip_left_right)




def show_imgs(img_list,gray_mode=None,titles=None,img_name=None,fontsize = 5,axis=0):
    if gray_mode is None :
        print(f" please give the gray mode ")
        return 

    num = len(img_list)
    for i in range(num):
        # ax = plt.subplot(1,num,i+1)
        if axis == 0:
            plt.subplot(1,num,i+1)
        elif axis ==1 : 
            plt.subplot(num,1,i+1)

        # ax.legend(..., fontsize=1)
        
        plt.imshow(img_list[i],cmap='gray' if gray_mode[i] else None)
        if titles is not None:
            # plt.title(titles[i],y=0.3,fontsize=5)
            plt.title(titles[i],fontsize=fontsize) #* y is relative to the image center? or the image bottom 
        plt.xticks([])
        plt.yticks([])
    if img_name is not None:
        plt.savefig(img_name,bbox_inches='tight')
    else:
        plt.show()
    
def imread(path,gray=False):
    if gray:
        return cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    return cv2.imread(path,cv2.IMREAD_COLOR)




def imread_PIL(path,gray=False):
    if gray:
        return Image.open(path).convert('L')   

    return Image.open(path)



'''
description:  detect  the edge in before_detect
param {*} before_detect
param {*} rgb_image
param {*} low_threshold
param {*} high_threshold
return {*}
'''
def detect_edge(src,low_threshold=50,high_threshold=150):

    
    if  len(src.shape)==2:
        #* enhance image 
        src = src[...,None]
        src = np.concatenate([src,src,src],axis=-1)
    
    if src.max()<=1:
        src = np.array(src*255,dtype=np.uint8)
        
    # src=cv2.GaussianBlur(src, (3, 3), 0)

    return cv2.Canny( src,low_threshold,high_threshold)


'''
description: interpolate image 
param {*} image
param {*} factor
return {*}
'''
def interpolate_image(image,factor=2):
    #* 1. move the  channel dimension to last chanel and transfer to torch tensor 
    if len(image.shape)==3:
        image = torch.from_numpy(image).permute(2,0,1)[None,...]
    else:
        image = torch.from_numpy(image)[None,None,...]

    #* 2. interpolate 
    image = F.interpolate(image,scale_factor=factor,)

    #* 3. transfer back  
    return image.squeeze().permute(1,2,0).numpy()









'''
description:  crop image through the center  
param {*} src
param {*} H
param {*} W
return {*}
'''
def crop_img(src,H,W):
    center_h, center_w=np.array(src.shape[:2],dtype=np.int32)/2
    cropped = src[int(center_h-H/2):int(center_h+H/2), int(center_w-W/2):int(center_w+W/2)]  # 裁剪坐标为[y0:y1, x0:x1]
    return cropped



'''
description: change one channel image to three channel. e.g. depth image 
param {*} img
return {*}
'''
def to_3_channel(img):
        img= img[...,None]
        return np.concatenate([img,img,img],axis=-1)
    
def merge_images(img1,alpha,img2,beta):
    if img1 is None or img2 is None:
        logger.info('image can not be None')
        
    
    def to_image(img):
        if img.max()<=1:
            return np.array(img*255,dtype=np.uint8)
        
        return img.astype(np.uint8)
    if img1.shape != img2.shape:
        if len(img1.shape) < len(img2.shape):
            #* assume the small one is gray scale 
            img1=to_3_channel(img1)
        else:
            img2=to_3_channel(img2)
            
    # elif img1.shape == img2.shape and len(img2.shape):
    #     img2=to_3_channel(img2)
    #     img1=to_3_channel(img1)
        
        
    img1=to_image(img1)
    img2=to_image(img2)

    return cv2.addWeighted(img1,alpha,img2,beta,0)

    # sample['image'],0.5,cv2.cvtColor(sample['edge']*255,cv2.COLOR_GRAY2RGB)

    


'''
description: 
param {*} predictions
param {*} filenames
param {*} output_dir
param {*} palettes
return {*}
'''
def save_colorful_images(predictions, filenames, output_dir, palettes):
   """
   Saves a given (B x C x H x W) into an image file.
   If given a mini-batch tensor, will save the tensor as a grid of images.
   """
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)





'''
description:  保存图像
param {*} predictions
param {*} filenames
param {*} output_dir
return {*}
'''
def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)





def shapen_img(src):

    # blur_img = cv.GaussianBlur(src, (0, 0), 5)
    blur_img = cv2.GaussianBlur(src, (3, 3),0)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)

    #cv.addWeighted(图1,权重1, 图2, 权重2, gamma修正系数, dst可选参数, dtype可选参数)
    return usm

def shapen_img2(src):


    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    dst = cv2.filter2D(src, -1, kernel=kernel)

    return dst


def shapen_Laplacian(in_img):
    I = in_img.copy()
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    L = cv2.filter2D(I, -1, kernel)
    a = 0.5
    O = cv2.addWeighted(I, 1, L, a, 0)
    O[O > 255] = 255
    O[O < 0] = 0
    return O



def shapen_USM(I):
    sigma = 3
    kernel_size = (0,0)
    L = cv2.GaussianBlur(I, kernel_size, sigma)
    L = L.astype(np.int32)
    I = I.astype(np.int32)
    H = I - L
    a = 1.5
    O = I + a*H
    O[O > 255] = 255
    O[O < 0] = 0
    O = O.astype(np.uint8)
    return O

def normalize(map):
    return (map - map.min())/(map.max() - map.min())


def normal_reverse_process(normals):
    #* from [-1,1] to [0,1]
    return (normals+1)/2 


def shapen_manual(src,band_width=3):
    SRC = np.fft.fft2(src)
    SRC = np.fft.fftshift(SRC)
    
    H = np.ones_like(SRC)
    

    
    H[int(H.shape[0]/2)-band_width:int(H.shape[0]/2)+band_width, int(H.shape[1]/2)-band_width:int(H.shape[1]/2)+band_width] = 0

    Y = SRC * H
    Y = np.fft.ifftshift(Y)
    y = np.fft.ifft2(Y)
    y = np.real(y)


    y= normalize(y)
    # show_imgs([y,SRC,src],[1,1,1],['after shapen','after frontier','origin image'])
    # show_imgs([y,src],[1,1],['after shapen','origin image'])
    return y




'''
description:  get the neighbors at (row_idx,col_idx) from map 
param {*} row_idx
param {*} col_idx
param {*} map
return {*}
'''
def get_neighbors_3_3(row_idx,col_idx,map):
    
    H,W =  map.shape[:2]
    need =np.array( [
            [[-1,-1], [0,-1],[1,-1]],
            [[-1,0], [0,0],[1,0]],
            [[-1,1], [0,1],[1,1]],
            ])

    # need =np.array( [
    #         [[-2,-2],[-1,-2], [0,-2],[1,-2],[2,-2]],
    #         [[-2,-1],[-1,-1], [0,-1],[1,-1],[2,-1]],
    #         [[-2,0],[-1,0], [0,0],[1,0],[2,0]],
    #         [[-2,1],[-1,1], [0,1],[1,1],[2,1]],
    #         [[-2,2],[-1,2], [0,2],[1,2],[2,2]],
    #         ])


    # print(map.shape)
    ans= [ ]
    map_size = need.shape
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            tmp = (row_idx,col_idx)+need[i][j]
            if tmp[0] >= 0 and tmp[1]>=0 and tmp[0] <H and tmp[1] <W:
                ans.append(map[tmp[0],tmp[1]])
            # else:
            #     print(tmp[0],tmp[1])

    return np.array(ans)
            




'''
description:  get the neighbors at (row_idx,col_idx) from map 
param {*} row_idx
param {*} col_idx
param {*} map
return {*}
'''
def get_neighbors_5_5(row_idx,col_idx,map):
    
    H,W =  map.shape[:2]

    need =np.array( [
            [[-2,-2],[-1,-2], [0,-2],[1,-2],[2,-2]],
            [[-2,-1],[-1,-1], [0,-1],[1,-1],[2,-1]],
            [[-2,0],[-1,0], [0,0],[1,0],[2,0]],
            [[-2,1],[-1,1], [0,1],[1,1],[2,1]],
            [[-2,2],[-1,2], [0,2],[1,2],[2,2]],
            ])


    # print(map.shape)
    ans= [ ]
    map_size = need.shape
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            tmp = (row_idx,col_idx)+need[i][j]
            if tmp[0] >= 0 and tmp[1]>=0 and tmp[0] <H and tmp[1] <W:
                ans.append(map[tmp[0],tmp[1]])
            # else:
            #     print(tmp[0],tmp[1])

    return np.array(ans)
            

'''
description: 
param {*} label_map
return {*}
'''
def get_edge_map_from_label(label_map):
    H,W=label_map.shape[:2]
    edge_map = np.zeros([H,W])
    for i in range(H):
        for j in range(W):
            neighbors = get_neighbors_3_3(i,j,label_map)
            if len(np.unique(neighbors))>1:
                edge_map[i,j]=1
    return edge_map
            
            
def imwrite(path,image):

    cv2.imwrite(path,image)   
          
def distance(a,b):
    return np.sqrt((a - b)**2)


def get_depth_edge_by_edge(depth_map,edge,depth_threshold):

    depth_edge = np.zeros(edge.shape[:2])
    idx=np.argwhere(edge==255)
    # print(f"edge pixel number :{idx.shape}")
    for r,c in idx:
        # neighbors = torch.from_numpy(get_neighbors(r,c,depth_map))   
        # dist = F.mse_loss(neighbors,torch.from_numpy(np.array(depth_map[r,c])).repeat([neighbors.shape[0]]),reduction='none')
        neighbors = get_neighbors_5_5(r,c,depth_map)
        
        dist = distance(neighbors,depth_map[r,c])
        dist= dist.max()
        if dist> depth_threshold:
            depth_edge[r,c]=1

    return depth_edge


# F.interpolate
def get_normal_depth_edge(depth_map,normal_map,edge_map=None,normals_threshold= None,depth_threshold= None):
    if normals_threshold is None or depth_threshold is None :
        logger.info(f" threshold can not be None ")
        return 


    H,W=depth_map.shape
    depth_edge = np.zeros([H,W])
    normal_edge = np.zeros([H,W])
    
    if edge_map is not None :
        for i in range(H):
            for j in range(W):
                if edge_map[i,j] == 1:
                    normal_neighbors = get_neighbors_3_3(i,j,normal_map)
                    depth_neighbors = get_neighbors_3_3(i,j,depth_map)
                    
                    # dist = F.mse_loss(normal_neighbors,torch.from_numpy(normal_map[i,j]).repeat([normal_neighbors.shape[0],1]),reduction='mean')
                    dist=1-F.cosine_similarity(
                                    normal_neighbors,
                                    torch.from_numpy(normal_map[i,j]).repeat([normal_neighbors.shape[0],1]),
                                    dim=-1)

                    dist = dist.max().numpy().tolist()


                    if dist> normals_threshold:
                        normal_edge[i,j]=1
                    
                    dist = F.mse_loss(depth_neighbors,
                                        torch.from_numpy(np.array(depth_map[i,j])).repeat([normal_neighbors.shape[0]]),
                                        reduction='none')
                    
                    # print('depth distance : ',depth_dist,'normal distance : ',normals_dist)
                    if dist.max().numpy().tolist()> depth_threshold:
                        depth_edge[i,j] = 1
    else :
        for i in range(H):
            for j in range(W):

                normal_neighbors = torch.from_numpy(get_neighbors_3_3(i,j,normal_map))
                depth_neighbors = torch.from_numpy(get_neighbors_3_3(i,j,depth_map))

                dist=1-F.cosine_similarity(
                                normal_neighbors,
                                torch.from_numpy(normal_map[i,j]).repeat([normal_neighbors.shape[0],1]),
                                dim=-1)

                dist = dist.max().numpy().tolist()


                if dist> normals_threshold:
                    normal_edge[i,j]=1
                
                dist = F.mse_loss(depth_neighbors,
                                    torch.from_numpy(np.array(depth_map[i,j])).repeat([normal_neighbors.shape[0]]),
                                    reduction='none')
                # print('depth distance : ',depth_dist,'normal distance : ',normals_dist)
                if dist.max().numpy().tolist()> depth_threshold:
                    depth_edge[i,j] = 1
        
    return depth_edge,normal_edge


if __name__ == "__main__":

    pass