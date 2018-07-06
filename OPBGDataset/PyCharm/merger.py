import os
import shutil

COREG_DIR = 'DIPG_COREG'

SEG_IN = 'DIPG_SEG'

coregs = {
    coreg_file.name.strip('.nii.gz'): coreg_file.path
    for coreg_file in os.scandir(SEG_IN) if coreg_file.is_file()
}

for patient_dir in os.scandir(COREG_DIR):
    if patient_dir.is_dir():
        shutil.copy(coregs[patient_dir.name], os.path.join(patient_dir.path, f'{patient_dir.name}_seg.nii.gz'))
