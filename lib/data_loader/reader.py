import os
import dicom
import cv2
import csv
import numpy as np
import math
from matplotlib import pyplot as plt
from path_manager import PATH

class Reader:
    def __init__(self,id):
        fname = PATH.raw_info
        with open(fname, 'r') as csvfile:
            rows = csv.reader(csvfile)
            self.id = id
            for row in rows:
                if row[0] == str(self.id):
                    self.name = row[1]
                    self.x = int(row[2])
                    self.y = int(row[3])
                    self.z = int(row[4])
                    self.loc = row[5]
        loc = self.loc
        print(self.name)
        ds = dicom.read_file(os.path.join(loc, 'dicomdir'))
        pixel_data = [np.zeros((250, 512, 512)), np.zeros((250, 512, 512))]

        for record in ds.DirectoryRecordSequence:
            if record.DirectoryRecordType == "IMAGE":
                # Extract the relative path to the DICOM file
                path = os.path.join(loc, record.ReferencedFileID[0], record.ReferencedFileID[1])
                dcm = dicom.read_file(path)
                if "WB" in dcm[0x08, 0x103e].value:
                    if "PET" in dcm[0x08, 0x103e].value:
                        pet_spa = dcm.PixelSpacing[0]
                        z = dcm.InstanceNumber
                        pet_len = dcm.Rows
                        slice_num = dcm.NumberOfSlices
                        pixel_data[0][z, 0:pet_len, 0:pet_len] = dcm.pixel_array
                    if "CT" in dcm[0x08, 0x103e].value:
                        ct_len = dcm.Rows
                        z = dcm.InstanceNumber
                        ct_spa = dcm.PixelSpacing[0]
                        pixel_data[1][z, 0:ct_len, 0:ct_len] = dcm.pixel_array

                if "5mm" in dcm[0x08, 0x103e].value:
                    ct_len = dcm.Rows
                    z = dcm.InstanceNumber
                    ct_spa = dcm.PixelSpacing[0]
                    pixel_data[1][z, 0:ct_len, 0:ct_len] = dcm.pixel_array

        pet0 = pixel_data[0][0:slice_num, 0:pet_len, 0:pet_len]
        pet1 = []
        pet2 = []
        ct = pixel_data[1][0:slice_num, 0:ct_len, 0:ct_len]
        del pixel_data
        for i in range(slice_num):
            pet1.append(cv2.resize(pet0[i], dsize=None, fx=pet_spa / ct_spa, fy=pet_spa / ct_spa))
        pet_len = pet1[0].shape[1]
        if (pet_len > 511):
            centre = int(pet_len / 2)
            for i in range(slice_num):
                pet2.append(pet1[i][centre - 256:centre + 256, centre - 256:centre + 256])
        else:
            len0 = pet_len
            if (len0 % 2 == 0):
                pad = int(256 - len0 / 2)
                for i in range(slice_num):
                    pet2.append(cv2.copyMakeBorder(pet1[i], pad, pad, pad, pad, cv2.BORDER_REPLICATE))
            else:
                pad = int(256 - len0 / 2)
                for i in range(slice_num):
                    pet2.append(cv2.copyMakeBorder(pet1[i], pad + 1 , pad, pad, pad + 1 , cv2.BORDER_REPLICATE))
        del pet1,pet0
        self.pet = np.array(pet2)
        print(self.pet.shape)
        self.ct = np.array(ct)
        del pet2,ct
        # plt.figure(figsize=(15, 15))
        # plt.subplot(2, 2, 1)
        # plt.imshow(pet[33])
        # plt.subplot(2, 2, 2)
        # plt.imshow(ct[33])
        # plt.subplot(2, 2, 3)
        # plt.imshow(ct[33], plt.cm.gray)
        # plt.imshow(pet[33], plt.cm.afmhot, alpha=0.5)
        # plt.scatter(250, 277, color='', marker='o', edgecolors='w', s=200)
        # plt.title(1)
        # plt.show()

    def crop(self, crop_xy, crop_z):
        try:
            pet_crop = self.pet[self.z-int(crop_z/2):self.z+int(math.ceil(crop_z/2.0)),self.x-int(crop_xy/2):self.x+int(math.ceil(crop_xy/2.0)),self.y-int(crop_xy/2):self.y+int(math.ceil(crop_xy/2.0))]
            ct_crop = self.ct[self.z-int(crop_z/2):self.z+int(math.ceil(crop_z/2.0))][self.x-int(crop_xy/2):self.x+int(math.ceil(crop_xy/2.0)),self.y-int(crop_xy/2):self.y+int(math.ceil(crop_xy/2.0))]
        except:
            print(str(crop_xy) + ' or ' + str(crop_z) + " is too large for " + str(self.id))
            return(0,0)
        else:
            return (pet_crop, ct_crop)

    def save(self):
        np.savez(os.path.join(PATH.npz_save,str(self.id)),pet = self.pet, ct = self.ct)

def  hello():
    print(1)