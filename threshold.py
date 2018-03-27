import dicom
rt = dicom.read_file("D:\doc\guo\diannei\data\biaozhu\DICOMDIR")
rcs = rt.ROIContourSequence
contour = rcs[i].ContourSequence[j].ContourData