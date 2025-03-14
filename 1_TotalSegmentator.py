"""
ENV.: Total Segmentator 
Run TotalSegmentator on venouse CT's from UKDD and PanCaim Manual_lables 


/mnt/cluster/environments/willmstid/UKDD/Radiomics/CT/PDAC/Venouse_Phase_POPF

/mnt/cluster/environments/willmstid/PANCAIM_CT_PDAC/raw_PanCaim/panorama_labels-main/manual_labels
/mnt/cluster/environments/willmstid/PANCAIM_CT_PDAC/raw_PanCaim/CT_scans

Compare segmentation with PanCaim Manual_lables 


ADD rescaling image for which are not in HU range
"""


import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from pathlib import Path
import tempfile
import pydicom
import logging
import warnings
from tqdm import tqdm

"""
Save CT-Scans as nifti files .nii.gz

"""
#dir_ven_POPF = "/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/ct_new_need_to_be_segmented_and_transformed_to_nii/new"
dir_ven_POPF = "/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/DICOM_original"

def collect_file_paths_in_leaf_directories(root_dir):
    """
    Collect paths to all files located in leaf directories within the given root directory.
    A leaf directory is defined as a directory that does not contain any subdirectories.
    """
    root_dir = Path(root_dir)
    file_paths = []


    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:  
            for filename in tqdm(filenames):
                file_paths.append(str(Path(dirpath) / filename))
    
    return file_paths

UKD_ven_POPF = collect_file_paths_in_leaf_directories(dir_ven_POPF)


directory_paths = [os.path.dirname(path) for path in UKD_ven_POPF]
unique_directory_paths = list(set(directory_paths))


def get_tag_value(dicom_dataset, tag_name, default=''):
    return getattr(dicom_dataset, tag_name, default) if tag_name in dicom_dataset else default

processed_list = []
error = []
save_path ="/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/ct_new_need_to_be_segmented_and_transformed_to_nii/nii_format/"

def save_as_nii(paths):
    
    processed_dict = {
        "saved": None, 
        "Image_name" : None
        }
    for img_path in tqdm(paths):
        try:
            reader = sitk.ImageSeriesReader()
            dicom_files = reader.GetGDCMSeriesFileNames(img_path)
            reader.SetFileNames(dicom_files)
            image = reader.Execute()
            dicom = pydicom.dcmread(dicom_files[0])
            output_filename = get_tag_value(dicom, "PatientID") + "_" + get_tag_value(dicom, "AcquisitionDate") + ".nii.gz"
            nii_img_name = os.listdir(save_path)
            
            if output_filename in nii_img_name:
                output_filename2 = "2_" + get_tag_value(dicom, "PatientID") + "_" + get_tag_value(dicom, "AcquisitionDate")  + ".nii.gz"
                print(output_filename2)
                sitk.WriteImage(image, os.path.join(save_path, output_filename2) )
                processed_dict = {"saved": 1, "Image_name": output_filename2 }
                processed_list.append(processed_dict)
            else :
                print(output_filename)
                sitk.WriteImage(image, os.path.join(save_path, output_filename))
                processed_dict = {"saved": 1, "Image_name": output_filename}
                processed_list.append(processed_dict)
    
        except Exception as e:
            error_stg = str(e)
            error.append((img_path,error_stg))
            processed_dict = {"saved": 0, "Image_name": output_filename}
            print(processed_dict)
            processed_list.append(processed_dict)
            continue
    
    return processed_list
    
processed_list = save_as_nii(unique_directory_paths)
    


"""
apply totalSegmentator
"""


import os
import nibabel as nib
from tqdm import tqdm
from totalsegmentator.python_api import totalsegmentator
import torch

print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.device_count())  # Number of GPUs
print(torch.cuda.get_device_name(0)) 


ven_ct_path = "/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/ct_new_need_to_be_segmented_and_transformed_to_nii/nii_format"
save_path_totalseg_total = "/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/ct_new_need_to_be_segmented_and_transformed_to_nii/TotalSegmentator/total"
save_path_totalseg_tissue_types = "/mnt/cluster/environments/willmstid/Projekte/POPF_Prediction/Radiomics/Image/CT_image/venous_phase/ct_new_need_to_be_segmented_and_transformed_to_nii/TotalSegmentator/tissue"


ven_cts_finish = os.listdir(save_path_totalseg_total)
ven_cts = os.listdir(ven_ct_path)

error_TotalSeg = []
for img_name in tqdm(ven_cts):
    
    try: 
        
        if img_name in ven_cts_finish:
            print(img_name)
            print("already done")
            continue
        
           
        else:
            print(img_name)
            print("Segment with TotalSegmentator")
            segmentation_result = totalsegmentator(os.path.join(ven_ct_path, img_name), ml=True, device='gpu', task="total") 
            nib.save(segmentation_result, os.path.join(save_path_totalseg_total, img_name ))
            
            segmentation_result = totalsegmentator(os.path.join(ven_ct_path, img_name), ml=True, device='gpu', task="tissue_types") 
            nib.save(segmentation_result, os.path.join(save_path_totalseg_tissue_types, img_name ))
    except Exception as e:
            error_msg = str(e)
            print(f"Error encountered: {error_msg}")
            error_TotalSeg.append((img_name,error_msg))
            continue




"""
apply totalSegmentator on PanCaim... 
copy to 

"""
import os
import shutil

path_manual_labels = "/mnt/cluster/environments/willmstid/PANCAIM_CT_PDAC/raw_PanCaim/panorama_labels-main/manual_labels"
path_TotalSeg = "/mnt/cluster/environments/willmstid/PANCAIM_CT_PDAC/TotalSegmentator_Pancreas/Totalsegmentator_all_v2"


destination_dir =  "/mnt/cluster/environments/willmstid/UKDD/Radiomics/CT/PDAC/Venouse_Phase_POPF/Pancaim_TotalSeg/TotalSeg"


manual_labels = os.listdir(path_manual_labels)
Total_Seg = os.listdir(path_TotalSeg)


error_TotalSeg = []
for manual_label in tqdm(manual_labels):
    
    
    
    try: 
        ct_name = manual_label.split(".")[0] + "_0000" + ".nii.gz"
        shutil.copy(os.path.join(path_TotalSeg, ct_name), destination_dir)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error encountered: {error_msg}")
        error_TotalSeg.append((ct_name,error_msg))
        continue







