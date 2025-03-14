"""
1.: resampling to same voxel size. 
    interpolation via BSpline 
2.: 

Resampling and Resizing of images via: 
 LanczosWindowedSinc (resize/non-resize)
 BSpline             (resize/non-resize)

TotalSegmentator Total
1 	spleen 	
2 	kidney_right 	
3 	kidney_left 	
4 	gallbladder 	
5 	liver 	
6 	stomach 	
7 	pancreas
15 	esophagus
52 	aorta
63 	inferior_vena_cava
64 	portal_vein_and_splenic_vein 	hepatic portal vein



TotalSegmentator Tissue
  "tissue_types": {
        1: "subcutaneous_fat",
        2: "torso_fat",
        3: "skeletal_muscle"
"""

import os
import SimpleITK as sitk
import numpy as np
import napari
viewer = napari.Viewer()
import cv2
from tqdm import tqdm
import pandas as pd
import ast
from joblib import Parallel, delayed
from scipy import ndimage

"""
1.:  get new voxel size (from other script)
2.:  direction alignment 
3.:  
"""




"""
get_larges_volume gives you the largest coherent organ because segmentation results give rise to smaller false segmented objects 
"""


def get_larges_volume(label, ROI):

    """
    Input:
        Lable
        ROI: threshold for segmented organ of interest 
    Output:
        ROI_mask: ROI 3D array as binary mask
    """
   
    binary_image = sitk.BinaryThreshold(label, lowerThreshold=ROI, upperThreshold=ROI, insideValue=1, outsideValue=0)
    connected_components = sitk.ConnectedComponent(binary_image)
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(connected_components)
    largest_label = None
    max_size = 0

    for label in shape_stats.GetLabels():
        size = shape_stats.GetNumberOfPixels(label)
        if size > max_size:
            max_size = size
            largest_label = label

    largest_component = sitk.BinaryThreshold(connected_components, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)
    ROI_mask = sitk.GetArrayFromImage(largest_component) 
    ROI_mask = ROI_mask * ROI
    return ROI_mask



"""
direction_allignment corrects image direction with the aid of organ orientation because the metadata can be wrong
Orientation:
    left: Liver = 5
    right: Spleen= 1
    top: vertebra T12 = 32
    down: vertebra  = 29
"""


def direction_alignment(img_name, img, label):
    """
    Input:
        img_name=str img name,
        img= sitk img,
        label= sitk label
        
    Output:
        
    """ 
    direction_ct = img.GetDirection() 
    direction_ct = np.array(direction_ct).reshape((3,3))
    
    superior = 32 #T12
    inferior = 29 #L3
    inferiro_2 = 30 #L2
    # check right-left direction with liver(5) -> spleen (1) (patient) (image: left-right)
    left = 5   #liver
    right = 1  #spleen
    
    label_np = sitk.GetArrayFromImage(label) 
    label_np = label_np.astype(np.uint8)
    labels = set(np.unique(label_np))

    if inferior not in labels:
        print(f"Label {inferior} not present")
        inferior = inferiro_2
        
    else:
        inferior = inferior
        
    labels_orientation = {inferior, superior, left, right}

    if labels_orientation.issubset(labels):
        superior_np = get_larges_volume(label, superior)
        inferior_np = get_larges_volume(label, inferior)
        right_np = get_larges_volume(label, right)
        left_np = get_larges_volume(label, left)
        error=None
    
        for z in range(inferior_np.shape[0]):  # interate through z-stack from bottom to top
            if inferior in inferior_np[z, :, :]:
                inferior_slice = z
                break
        
        for z in range(superior_np.shape[0]-1, -1, -1):  #interate through z-stack from top to bottom
            if superior in superior_np[z,:, :]:
                superior_slice = z 
                break
                
        
        for x in range(right_np.shape[2]-1, -1, -1):  # interate through z-stack right to left
            if right in right_np[:, :, x]:
                right_slice = x
                break
        
             
        for x in range(left_np.shape[2]):  #interate through  left to right
            if left in left_np[:,:,x]:
                left_slice = x
                break
            
        if (superior_slice > inferior_slice and left_slice < right_slice):   #00
            print("Image in inferior-superior and left-right direction")
        
        elif (superior_slice < inferior_slice and left_slice > right_slice): #11
            print("Correcting to inferior-superior and left-right direction")
            #img_direction: new direction to align image 
            img_direction = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
            img_direction_matrix = np.array(img_direction).reshape((3,3))
                    
            flip_axes = direction_ct.diagonal() != img_direction_matrix.diagonal()
            flip_axes = flip_axes.tolist()
            flip_filter = sitk.FlipImageFilter()
            flip_filter.SetFlipAxes(flip_axes)
                    
            img = flip_filter.Execute(img)
            label = flip_filter.Execute(label)
        
        elif (superior_slice < inferior_slice and left_slice < right_slice): #10
            print("Correcting to inferior-superior")
            img_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
            img_direction_matrix = np.array(img_direction).reshape((3,3))
                    
            flip_axes = direction_ct.diagonal() != img_direction_matrix.diagonal()
            flip_axes = flip_axes.tolist()
            flip_filter = sitk.FlipImageFilter()
            flip_filter.SetFlipAxes(flip_axes)
                    
            img = flip_filter.Execute(img)
            label = flip_filter.Execute(label)
            
        elif (superior_slice > inferior_slice and left_slice > right_slice): #01
            print("Correcting to left-right")
            
            img_direction = (-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
            img_direction_matrix = np.array(img_direction).reshape((3,3))
                    
            flip_axes = direction_ct.diagonal() != img_direction_matrix.diagonal()
            flip_axes = flip_axes.tolist()
            flip_filter = sitk.FlipImageFilter()
            flip_filter.SetFlipAxes(flip_axes)
                    
            img = flip_filter.Execute(img)
            label = flip_filter.Execute(label)
    
           
    else:
        missing_values = labels_orientation - labels
        print(f"Missing segmentation: {missing_values}")
        error = missing_values
    
    direction = [img.GetDirection()]
    print(f"img and label are orientated to: {direction}")
    
    return img, label, direction, error

 
def resampling_and_resizing(ct_img, seg_total, seg_tissue, new_voxel_size, BSpline=True, Lanczos=False, resize=False):
    """
        Input:
            img: CT-scan (.nii file )
            seg_total: int array with labels 
            seg_tissue: int array with labels 
            new_voxel_size: median voxel size over all images 
            resize: default=False (enables voxel size harmonisation, if False)
            BSpline: default=True
            Lanczos: Defailt=False
    """
    #Calculate new img dimension for rescaling
    voxel_size          = ct_img.GetSpacing()     #gets original voxel size
    img_dim             = ct_img.GetSize()        #gets img dimension
    patient_orientation = ct_img.GetOrigin()      #gets patient orientation in scanner(-208.62960815429688, -164.68431091308594, 1912.5)
    img_orientation     = ct_img.GetDirection()   #gets patient and detector orientation(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    
    #new image dim to 
    new_img_dim = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(img_dim, voxel_size, new_voxel_size)]
    
    

    resample = sitk.ResampleImageFilter()                       #initalize resampling class as resample
    resample.SetOutputSpacing(new_voxel_size)                   #Set new voxel size
    if resize:
       resample.SetSize(img_dim)                                #Set image dimentions, if resize=True -> original dim
          
    else:
       resample.SetSize(new_img_dim)                           #Set image dimentions, else new image dim, no resizing 
         
            
    resample.SetOutputDirection(img_orientation)                #sets patient and detector orientation
    resample.SetOutputOrigin(patient_orientation)               #gets patient orientation in scanner
    resample.SetTransform(sitk.Transform())                     #sitk.Transform() Identity transformation, no translation, rotation, ect.
    resample.SetDefaultPixelValue(-1024)        #2 SetDefaultPixelValue(itk_image.GetPixelIDValue()) 
    if Lanczos:                                                 #Set Interploator 
        resample.SetInterpolator(sitk.sitkLanczosWindowedSinc)      
    elif BSpline:
        resample.SetInterpolator(sitk.sitkBSpline) 
    img_resample = resample.Execute(ct_img)                        #Execute Interpolator  


#########################################################################################################################################
#Resample Label via  sitkNearestNeighbor 
                               
    resample = sitk.ResampleImageFilter()                       #initalize resampling
    resample.SetOutputSpacing(new_voxel_size)                   #Set new voxel size
    if resize:
       resample.SetSize(img_dim)                                #Set image dimentions, if resize=True -> original dim
          
    else:
       resample.SetSize(new_img_dim)                            #else no resize to original dim 
                                                                #new img dim due to new voxel size
    resample.SetOutputDirection(img_orientation)                #sets patient and detector orientation
    resample.SetOutputOrigin(patient_orientation)               #gets patient orientation in scanner
    resample.SetTransform(sitk.Transform())                     #sitk.Transform() Identity transformation, no translation, rotation, ect.
    resample.SetDefaultPixelValue(-1024)        #2 SetDefaultPixelValue(itk_image.GetPixelIDValue())   also here or to lable 
    resample.SetInterpolator(sitk.sitkNearestNeighbor)          #Set Interploator for labels -> nearest Neighbor
    
    seg_total_resample = resample.Execute(seg_total)    
    seg_tissue_resample = resample.Execute(seg_tissue)    
    
    
    return img_resample, seg_total_resample, seg_tissue_resample



def crop_abdomen(img_np):
    #get foreground by thresholding 
    binary_mask = np.where(img_np >= -250, 1, 0)   #threshold between air and fat/soft tissue
    binary_mask = binary_mask.astype(np.uint8)
    mask_foreground = []
    mask_stack = []
    kernel = np.ones((3,3), np.uint8) 
    kernel_erode = np.ones((3,3), np.uint8)
    for z in range(binary_mask.shape[0]):  
        img_slice = binary_mask[z, :, :]
        closing = cv2.morphologyEx(img_slice, cv2.MORPH_CLOSE, kernel) #fille holes
        #dilation = cv2.dilate(closing, kernel,iterations = 3)
        #opened_image = cv2.morphologyEx(img_slice, cv2.MORPH_OPEN, kernel_opening,iterations = 3)
        num_labels, labels_im = cv2.connectedComponents(closing) #instance segmentation
        max_label = 1
        max_size = 0
        for i in range(1, num_labels):  # getting biggest instance segmentation, hopefully only outer body part
            if np.sum(labels_im == i) > max_size:
                max_size = np.sum(labels_im == i)
                max_label = i
        body = np.where(labels_im == max_label, 1, 0) 
        contours, hierarchy = cv2.findContours(body, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled_contours = np.zeros_like(img_slice)
        for i in range(len(contours)):
            cv2.drawContours(filled_contours, contours, i, 1, thickness=cv2.FILLED)
            
        mask_slice = body + filled_contours  #one pixel overlay hence 2
        mask_slice = np.where(mask_slice >= 1, 1, 0)
        mask_slice = mask_slice.astype(np.uint8)
        eroded_mask = cv2.erode(mask_slice, kernel_erode, iterations=1)   
        eroded_mask = np.where(eroded_mask >= 1, 0, -5000)
        
        mask_foreground.append(mask_slice)
        mask_stack.append(eroded_mask) 
       
    mask_stack = np.array(mask_stack)
    mask_foreground = np.array(mask_foreground)
    
    masked_cropped_img = img_np + mask_stack
    masked_cropped_img[masked_cropped_img <= -1024] = -1024
    abdomen_foreground = masked_cropped_img
    return abdomen_foreground, mask_foreground     




def get_closing_and_dilation(mask_subfat_lining, iterations):
    
    kernel = np.ones((3,3), np.uint8)
    list_dilation_3d = []
    
    for z in range(mask_subfat_lining.shape[0]):
        mask_slice = mask_subfat_lining[z,:,:]
        closing = cv2.morphologyEx(mask_slice, cv2.MORPH_CLOSE, kernel) #fille holes
        dilation = cv2.dilate(closing, kernel,iterations = iterations)
        list_dilation_3d.append(dilation)
        
    return np.stack(list_dilation_3d, axis=0)


def get_fat_volume(img_name, ct_img, seg_total, seg_tissue):
          #ct_img = ct_img_resample
          #seg_total = seg_total_resample
          #seg_tissue = seg_tissue_resample
          
          PID = img_name.split("_")[0]
          ROI=7 #Pancreas = 7
               
          mask = get_larges_volume(seg_total, ROI)  
          mask = (mask == ROI)
          
          img_np = sitk.GetArrayFromImage(ct_img) 
          seg_tissue_np = sitk.GetArrayFromImage(seg_tissue).astype(np.uint8) 
          seg_total_np = sitk.GetArrayFromImage(seg_total).astype(np.uint8)  
     
####################################################################################################           
          #Crop images 
          
          z_indices, y_indices, x_indices = np.nonzero(mask)
          z_min, z_max = z_indices.min(), z_indices.max() 
          y_min, y_max = y_indices.min(), y_indices.max()
          x_min, x_max = x_indices.min(), x_indices.max() 
          
          z_delta = z_max-z_min
          y_delta = y_max-y_min
          x_delta = x_max-x_min
          
          bb_volume = z_delta*y_delta*x_delta  #Volume of bounding box
          
          z_max += 1
          y_max += 1
          x_max += 1

          ct_ROI            = img_np[z_min:z_max, :, :]
          mask_ROI          = mask[z_min:z_max, :, :]
          seg_tissue_np_roi = seg_tissue_np[z_min:z_max, :, :]
          seg_total_np_roi  = seg_total_np[z_min:z_max, :, :]
          
          mask_subfat = (seg_tissue_np_roi == 1).astype(int)
          mask_vicfat = (seg_tissue_np_roi == 2).astype(int)
          mask_muscle = (seg_tissue_np_roi == 3).astype(int)
          
####################################################################################################           
          #1: get relative volume of visceral fat normalized with body volume high pancreas
          # visceralfat/(body vol - subcultane fat)
          
          abdomen_foreground, mask_foreground = crop_abdomen(ct_ROI)
          
          abs_abdomen_vicfat =  np.sum(mask_vicfat)
          rel_body_vicfat = np.sum(mask_vicfat)/(np.sum(mask_foreground) - np.sum(mask_subfat)) 
          
          abs_body_subfat =  np.sum(mask_subfat)
          rel_body_subfat = np.sum(mask_subfat)/(np.sum(mask_foreground)) 
          
          abs_body_fat = np.sum(mask_subfat) + np.sum(mask_vicfat)
          rel_body_fat = (np.sum(mask_subfat) + np.sum(mask_vicfat))/np.sum(mask_foreground)
          
####################################################################################################           
          #1: get relative volume of visceral fat normalized with abdomen volume high pancreas
          #       
          
          #lung 10,11,12,13,14
          #vertebrae 28,29,30,31,32,33
          vertebrae_list = [28,29,30,31,32,33,10,11,12,13,14]  
          mask_vertebrae = np.isin(seg_total_np_roi, vertebrae_list).astype(np.uint8)

          mask_subfat_ROI_sitk = sitk.GetImageFromArray(mask_subfat)
          mask_muscle_ROI_sitk = sitk.GetImageFromArray(mask_muscle)
          mask_vertebrae_ROI_sitk = sitk.GetImageFromArray(mask_vertebrae)
          
          mask_subfat_lining = get_larges_volume(mask_subfat_ROI_sitk, ROI=1) 
          mask_muscel_lining = get_larges_volume(mask_muscle_ROI_sitk, ROI=1) 
          
          
          dilation_3d_mask_subfat_lining = get_closing_and_dilation(mask_subfat_lining, iterations=3)
          dilation_3d_mask_muscel_lining = get_closing_and_dilation(mask_muscel_lining, iterations=1)
          dilation_3d_mask_mask_vertebraeg = get_closing_and_dilation(mask_vertebrae, iterations=2)
          
          outside = dilation_3d_mask_subfat_lining + dilation_3d_mask_muscel_lining + dilation_3d_mask_mask_vertebraeg
          
          outside = (outside >= 1).astype(np.uint8)
          kernel = np.ones((9,9), np.uint8)
          outside_2 = cv2.morphologyEx(outside, cv2.MORPH_CLOSE, kernel)
          
          #fill holes slice wise
          filled3D = []
          kernel = np.ones((3,3), np.uint8) 
          for z in range(outside_2.shape[0]):
              slice_2d = outside_2[z,:,:]
              height, width = slice_2d.shape
              slice_2d[width // 2, :] = 0
              filled = ndimage.binary_fill_holes(slice_2d).astype(np.uint8)
              filled = cv2.dilate(filled, kernel,iterations = 1)
              filled = cv2.erode(filled, kernel, iterations=1)
              filled3D.append(filled)
              
          huelle = np.stack(filled3D, axis=0)
            
          abdomen_vol = np.sum(huelle)
          
          rel_abdomen_vicfat = np.sum(mask_vicfat) / abdomen_vol
          

#################################################################################################### 

          kernel = ndimage.generate_binary_structure(3, 1)

          mask_dilated_10 = ndimage.binary_dilation(mask_ROI, structure=kernel, iterations=10)
          mask_dilated_15 = ndimage.binary_dilation(mask_ROI, structure=kernel, iterations=15)
          mask_dilated_20 = ndimage.binary_dilation(mask_ROI, structure=kernel, iterations=20)
          mask_dilated_25 = ndimage.binary_dilation(mask_ROI, structure=kernel, iterations=25)
          mask_dilated_30 = ndimage.binary_dilation(mask_ROI, structure=kernel, iterations=30)
         
          
          dilated_10 = mask_dilated_10.astype(int) - mask_ROI.astype(int)
          dilated_15 = mask_dilated_15.astype(int) - mask_ROI.astype(int) 
          dilated_20 = mask_dilated_20.astype(int) - mask_ROI.astype(int) 
          dilated_25 = mask_dilated_25.astype(int) - mask_ROI.astype(int)
          dilated_30 = mask_dilated_30.astype(int) - mask_ROI.astype(int) 
          
          
          dilated_10_vol_vicfat =  np.sum(dilated_10 * mask_vicfat) / np.sum(dilated_10)
          dilated_15_vol_vicfat =  np.sum(dilated_15 * mask_vicfat) / np.sum(dilated_15)
          dilated_20_vol_vicfat =  np.sum(dilated_20 * mask_vicfat) / np.sum(dilated_20)
          dilated_25_vol_vicfat =  np.sum(dilated_25 * mask_vicfat) / np.sum(dilated_25)
          dilated_30_vol_vicfat =  np.sum(dilated_30 * mask_vicfat) / np.sum(dilated_30)
          
          #Get locations of non zero pixels
          
         
          #ct_ROI_bb = img_np[z_min:z_max, y_min:y_max, x_min:x_max]
          #mask_ROI_bb = mask[z_min:z_max, y_min:y_max, x_min:x_max]
          #mask_vicfat_ROI_bb = mask_vicfat[z_min:z_max, y_min:y_max, x_min:x_max]  
            
          pancreas_volume = np.sum(mask_ROI)
          #z_stack_volume = np.sum(mask_innerVolume)
          #subfat_volume = np.sum(mask_subfat_ROI)
          #vicfat_volume = np.sum(mask_vicfat_ROI)
          #vicfat_volume_bb = np.sum(mask_vicfat_ROI_bb)
          
          #rel_vicfat_abdomen= vicfat_volume / z_stack_volume 
          #rel_vicfat_bb= vicfat_volume_bb / bb_volume 



       
          dict_readouts = {
              "img_name" : img_name,
              "PID": PID,
              "Pancreas_Volume" : pancreas_volume, 
              "z_height": z_delta,
              "x_length": x_delta,
              "y_width": y_delta,
              "abdomen_vol":abdomen_vol,
              "abs_abdomen_vicfat" : abs_abdomen_vicfat,
              "rel_abdomen_vicfat": rel_abdomen_vicfat,
   
              "rel_body_vicfat" : rel_body_vicfat,
              "abs_body_subfat" : abs_body_subfat,
              "rel_body_subfat" : rel_body_subfat,
              "abs_body_fat" : abs_body_fat,
              "rel_body_fat":rel_body_fat,
              "rel_vicfat_10":dilated_10_vol_vicfat,
              "rel_vicfat_15":dilated_15_vol_vicfat,
              "rel_vicfat_20":dilated_20_vol_vicfat,
              "rel_vicfat_25":dilated_25_vol_vicfat,
              "rel_vicfat_30":dilated_30_vol_vicfat
              }
          
          return dict_readouts, ct_ROI, mask_ROI, mask_vicfat, mask_subfat, huelle, mask_dilated_30




"""            
seg_tissue_resample_np = sitk.GetArrayFromImage(seg_tissue_resample)   
#ct_ROI_np = sitk.GetArrayFromImage(ct_ROI)  
    
viewer = napari.Viewer()
viewer.add_image(ct_ROI, name='CT Scan')
viewer.add_image(dilation_3d_mask_subfat_lining, name='dilation_3d_mask_subfat_lining')
viewer.add_image(dilation_3d_mask_muscel_lining, name='dilation_3d_mask_muscel_lining')
viewer.add_image(mask_ROI, name='mask_ROI')
viewer.add_image(dilation_3d_mask_mask_vertebraeg, name='mask_vertebrae')
viewer.add_image(closing, name='closing')
"""        
"""
add 
mask_muscle and mask_subfat together and do closing and dilation only 1 it
seg_tissue_resample_np = sitk.GetArrayFromImage(seg_tissue_resample)   
#ct_ROI_np = sitk.GetArrayFromImage(ct_ROI)  
    
viewer = napari.Viewer()
viewer.add_image(ct_ROI, name='CT Scan')
viewer.add_image(mask_innerVolume, name='mask_innerVolume')
viewer.add_image(mask_subfat_ROI, name='mask_subfat_ROI')
viewer.add_image(mask_vicfat_ROI, name='mask_vicfat_ROI')
viewer.add_image(mask_dilation_3d, name='dilation_3d')
viewer.add_image(mask_ROI, name='pancres') 


viewer = napari.Viewer()
viewer.add_image(ct_ROI, name='CT Scan')
viewer.add_image(abdomen_foreground, name='abdomen_foreground')      
viewer.add_image(mask_foreground, name='mask_foreground')  
viewer.add_image(mask_subfat, name='mask_subfat') 
viewer.add_image(outside, name='outside') 
viewer.add_image(outside_2, name='outside_2') 
viewer.add_image(outside_3, name='outside_3') 
viewer.add_image(huelle, name='huelle') 
"""          
    

path_ct_scan = "/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/nibabel_format/"
path_seg_tissue = "/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/Segmentation_TotalSegmentator/tissue_types"
path_seg_total = "/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/Segmentation_TotalSegmentator/total"

ct_scans = os.listdir(path_ct_scan)  
seg_tissues = os.listdir(path_seg_tissue)
seg_total = os.listdir(path_seg_total)


new_voxel_size = [0.740, 0.740, 3.0]
list_error = []
list_dict_readouts = []      

for ct_name in ct_scans:
     
    try: 
         
    
         #ct_name = ct_scans[11]
         seg_total = sitk.ReadImage(os.path.join(path_seg_total, ct_name))
         seg_tissue = sitk.ReadImage(os.path.join(path_seg_tissue, ct_name))
         ct_img = sitk.ReadImage(os.path.join(path_ct_scan, ct_name))
     
         ct_img, seg_total, direction, error = direction_alignment(img_name=ct_name, img=ct_img, label=seg_total) 
         
         #cant orientate tissue, only with label from total 
         #direction: is the new direction for inferior-superior and left-right direction
         direction_matrix = np.array(direction).reshape((3,3))
         
         seg_tissue_direction = seg_tissue.GetDirection()
         seg_tissue_direction_matrix = np.array(seg_tissue_direction).reshape((3,3))
         
         flip_axes = direction_matrix.diagonal() != seg_tissue_direction_matrix.diagonal()
         flip_axes = flip_axes.tolist()
         flip_filter = sitk.FlipImageFilter()
         flip_filter.SetFlipAxes(flip_axes)
         seg_tissue = flip_filter.Execute(seg_tissue)
     
         #resampling and resizing 
         ct_img_resample, seg_total_resample, seg_tissue_resample = resampling_and_resizing(ct_img, seg_total, seg_tissue, new_voxel_size)
        
         dict_readouts, ct_ROI, mask_ROI, mask_vicfat_ROI, mask_subfat_ROI, ct_ROI_bb, mask_ROI_bb, mask_vicfat_ROI_bb = get_fat_volume(ct_name, ct_img_resample, seg_total_resample, seg_tissue_resample)
         
         ct_ROI = sitk.GetImageFromArray(ct_ROI)
         mask_ROI = sitk.GetImageFromArray(mask_ROI.astype(np.uint8))
         mask_vicfat_ROI = sitk.GetImageFromArray(mask_vicfat_ROI.astype(np.uint8))
         mask_subfat_ROI = sitk.GetImageFromArray(mask_subfat_ROI.astype(np.uint8))
         ct_ROI_bb = sitk.GetImageFromArray(ct_ROI_bb)
         mask_ROI_bb = sitk.GetImageFromArray(mask_ROI_bb.astype(np.uint8))
         mask_vicfat_ROI_bb = sitk.GetImageFromArray(mask_vicfat_ROI_bb.astype(np.uint8))
         
         
         sitk.WriteImage(ct_ROI, os.path.join("/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/output_loreen/CT_ROI/",ct_name))
         sitk.WriteImage(mask_ROI, os.path.join("/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/output_loreen/mask_ROI/",ct_name))
         sitk.WriteImage(mask_vicfat_ROI, os.path.join("/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/output_loreen/mask_viceral_fat/",ct_name))
         sitk.WriteImage(mask_subfat_ROI, os.path.join("/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/output_loreen/mask_subfat_ROI/",ct_name))
         sitk.WriteImage(ct_ROI_bb, os.path.join("/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/output_loreen/ct_ROI_bb/",ct_name))
         sitk.WriteImage(mask_ROI_bb, os.path.join("/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/output_loreen/mask_ROI_bb/",ct_name))
         sitk.WriteImage(mask_vicfat_ROI_bb, os.path.join("/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/output_loreen/mask_vicfat_ROI_bb/",ct_name))
    
         
         list_dict_readouts.append(dict_readouts)
     
    except Exception as e:
         error_msg = str(e)
         print(f"Error encountered: {error_msg}")
         
         dict_error = {
                     "CT_name" : [ct_name],
                     "CT_Error": [error_msg]}
     
     
         list_error.append((dict_error))
     
        
df_popf_fat = pd.DataFrame(list_dict_readouts) 
df_popf_fat_error =pd.DataFrame(list_error)


df_popf_fat.to_excel("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/df_popf_fat.xlsx")     
df_popf_fat_error.to_excel("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF//df_popf_fat_error.xlsx")
    

ct_img_resample_np = sitk.GetArrayFromImage(ct_img_resample) 
seg_total_resample_np = sitk.GetArrayFromImage(seg_total_resample) 
seg_tissue_resample_np = sitk.GetArrayFromImage(seg_tissue_resample)     
  

viewer = napari.Viewer()
viewer.add_image(ct_ROI, name='CT Scan')
viewer.add_image(mask_ROI, name='mask_ROI_np')
viewer.add_image(mask_vicfat_ROI, name='mask_vicfat_ROI_np')
viewer.add_image(mask_subfat_ROI, name='mask_subfat_ROI_np')




viewer = napari.Viewer()
viewer.add_image(ct_img_resample_np, name='CT Scan')
viewer.add_image(seg_total_resample_np, name='seg_tissue_np')
viewer.add_image(seg_tissue_resample_np, name='seg_total_np')


     
new_voxel_size = [1, 1, 1]  
list_error = []
list_error_alignment = []
list_dict_readouts = [] 
def process_scan(ct_name):
    
  
    try:
        ct_name = clean_list[0]
        seg_total = sitk.ReadImage(os.path.join(path_seg_total, ct_name))
        seg_tissue = sitk.ReadImage(os.path.join(path_seg_tissue, ct_name))
        ct_img = sitk.ReadImage(os.path.join(path_ct_scan, ct_name))
        ct_img, seg_total, direction, error = direction_alignment(img_name=ct_name, img=ct_img, label=seg_total) 
        list_error_alignment.append((ct_name,error,direction))
        #cant orientate tissue, only with label from total 
        #direction: is the new direction for inferior-superior and left-right direction
        direction_matrix = np.array(direction).reshape((3,3))
        
        seg_tissue_direction = seg_tissue.GetDirection()
        seg_tissue_direction_matrix = np.array(seg_tissue_direction).reshape((3,3))
        
        flip_axes = direction_matrix.diagonal() != seg_tissue_direction_matrix.diagonal()
        flip_axes = flip_axes.tolist()
        flip_filter = sitk.FlipImageFilter()
        flip_filter.SetFlipAxes(flip_axes)
        seg_tissue = flip_filter.Execute(seg_tissue)
    
        #resampling and resizing 
        ct_img_resample, seg_total_resample, seg_tissue_resample = resampling_and_resizing(ct_img, seg_total, seg_tissue, new_voxel_size)
        
        dict_readouts, ct_ROI, mask_ROI, mask_vicfat, mask_subfat, huelle, mask_dilated_30 = get_fat_volume(ct_name, ct_img_resample, seg_total_resample, seg_tissue_resample)
        
        ct_ROI = sitk.GetImageFromArray(ct_ROI)
        mask_ROI = sitk.GetImageFromArray(mask_ROI.astype(np.uint8))
        mask_vicfat = sitk.GetImageFromArray(mask_vicfat.astype(np.uint8))
        mask_subfat = sitk.GetImageFromArray(mask_subfat.astype(np.uint8))
        huelle = sitk.GetImageFromArray(huelle)
        mask_dilated_30 = sitk.GetImageFromArray(mask_dilated_30.astype(np.uint8))
             
        sitk.WriteImage(ct_ROI, os.path.join("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/CT_ROI",ct_name))
        sitk.WriteImage(mask_ROI, os.path.join("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/mask_ROI",ct_name))
        sitk.WriteImage(mask_vicfat, os.path.join("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/mask_vicfat",ct_name))
        sitk.WriteImage(mask_subfat, os.path.join("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/mask_subfat",ct_name))
        sitk.WriteImage(huelle, os.path.join("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/huelle",ct_name))
        sitk.WriteImage(mask_dilated_30, os.path.join("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/mask_dilated_30",ct_name))
       
   
        list_dict_readouts.append(dict_readouts)
            
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error encountered: {error_msg}")
        
        dict_error = {
                    "CT_name" : ct_name,
                    "Errror_": error_msg}
    
    
        list_error.append((dict_error))
  
    return list_error, list_dict_readouts, list_error_alignment

"""    
just the cohort for loreen
"""

viewer = napari.Viewer()
viewer.add_image(ct_ROI, name='CT Scan')
viewer.add_image(mask_innerVolume, name='mask_innerVolume')
viewer.add_image(mask_subfat_ROI, name='mask_subfat_ROI')
viewer.add_image(mask_vicfat_ROI, name='mask_vicfat_ROI')
viewer.add_image(mask_dilation_3d, name='dilation_3d')
viewer.add_image(mask_ROI, name='pancres') 



os.makedirs("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/CT_ROI", exist_ok=True)
os.makedirs("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/mask_ROI", exist_ok=True)
os.makedirs("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/mask_vicfat", exist_ok=True)
os.makedirs("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/mask_subfat", exist_ok=True)
os.makedirs("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/huelle", exist_ok=True)
os.makedirs("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/images/mask_dilated_30", exist_ok=True)

ct_scans    
df_loreen = pd.read_excel("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/df_popf_fat_date.xlsx")
ct_scans = df_loreen["img_name"].to_list() 
import math
clean_list = [f for f in ct_scans if not (isinstance(f, float) and math.isnan(f))]
    
results = Parallel(n_jobs=25)(delayed(process_scan)(ct_name) for ct_name in tqdm(clean_list))   
 
list_error, list_dict_readouts, list_error_alignment = zip(*results)
 
list_dict_readouts = [entry for sublist in list_dict_readouts for entry in sublist]
list_dict_erros = [entry for sublist in list_error for entry in sublist]
list_dict_erros_alignment = [entry for sublist in list_error_alignment for entry in sublist]

#new_voxel_size = [0.740, 0.740, 3.0]
  
voxel_vol = 1*1*1*0.001 #in mL 

df_popf_fat = pd.DataFrame(list_dict_readouts) 
df_popf_fat_error =pd.DataFrame(list_dict_erros)
df_popf_fat_error_alignment =pd.DataFrame(list_dict_erros_alignment)


cols = ['Pancreas_Volume', 'abdomen_vol', 'abs_abdomen_vicfat','abs_body_subfat', 'abs_body_fat']
df_popf_fat[cols] = (df_popf_fat[cols] * 0.001).astype(int)


#df_popf_fat["z_height"] = (df_popf_fat["z_height"] * 1).astype(int)

#cols = ['x_length', 'y_width']
#df_popf_fat[cols] = (df_popf_fat[cols] * 1)


df_popf_fat.to_excel("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/df_popf_fat_loreen.xlsx")     
df_popf_fat_error.to_excel("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/df_popf_fat_error_loreen.xlsx")
df_popf_fat_error_alignment.to_excel("/mnt/cluster/environments/willmstid/Projekte/Loreen_POPF/output/df_popf_fat_error_alignment_loreen.xlsx")
     
#df_popf_fat_error_alignment_v2 = df_popf_fat_error_alignment[df_popf_fat_error_alignment[1] != None]





