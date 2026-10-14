import os
import subprocess
import sys
import tempfile

# dcm2niix executable (override with $DCM2NIIX; defaults to PATH lookup)
DCM2NIIX_PATH = os.environ.get("DCM2NIIX", "dcm2niix")

def convert_dicoms_to_nifti(root_dir, output_base_dir):
    """
    Batch-convert DICOM files under the specified root directory to NIfTI format.

    Args:
        root_dir (str): path to the root directory containing multiple DICOM folders.
        output_base_dir (str): root directory where all converted .nii.gz files are stored.
    """

    # Ensure the output root directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Statistics
    total_success = 0
    total_fail = 0

    # Iterate over all subfolders under the root directory
    for item in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, item)
        if not os.path.isdir(folder_path):
            continue

        print(f"\nProcessing folder: {item}")

        # Check whether the folder contains DICOM files
        dcm_files = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.dcm', '.ima'))]
        if not dcm_files:
            print(f"  Skipping: no DICOM files found in folder '{item}'.")
            continue

        # Build output filename: foldername_0000.nii.gz
        output_filename = f"{item}_0000"
        output_path = os.path.join(output_base_dir, f"{output_filename}.nii.gz")
        if os.path.exists(output_path):
            print(f"  Skipping: output file already exists: {output_filename}.nii.gz")
            total_success += 1
            continue
        # Try direct conversion
        command = [
            DCM2NIIX_PATH,
            '-z', 'y',
            '-f', output_filename,
            '-o', output_base_dir,
            folder_path
        ]

        success = False
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"  ✅ Conversion complete: {output_filename}.nii.gz")
            if result.stdout:
                print(f"    Output: {result.stdout.strip()}")
            total_success += 1
            success = True
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️  Direct conversion failed (error: {e.stderr.strip()[:100] if e.stderr else 'unknown error'}...)")
            
            print(f"  ⚠️  Trying gdcmconv preprocessing...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                converted_files = []
                for dcm_file in dcm_files:
                    src_path = os.path.join(folder_path, dcm_file)
                    dst_path = os.path.join(temp_dir, dcm_file)
                    try:
                        subprocess.run(
                            ['gdcmconv', '--raw', src_path, dst_path],
                            capture_output=True,
                            check=True
                        )
                        converted_files.append(dst_path)
                    except subprocess.CalledProcessError as gdcm_err:
                        print(f"    ⚠️  gdcmconv failed to process {dcm_file}")
                
                if converted_files:
                    command_pre = [
                        DCM2NIIX_PATH,
                        '-z', 'y',
                        '-f', output_filename,
                        '-o', output_base_dir,
                        temp_dir
                    ]
                    try:
                        result = subprocess.run(command_pre, capture_output=True, text=True, check=True)
                        print(f"  ✅ Conversion complete after preprocessing: {output_filename}.nii.gz")
                        total_success += 1
                        success = True
                    except subprocess.CalledProcessError as e2:
                        print(f"  ❌ Still failed after preprocessing: {e2.stderr}")
                else:
                    print(f"  ❌ Preprocessing failed for all files")


        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")

        if not success:
            total_fail += 1

    print("\n" + "="*60)
    print(f"✅ Conversion complete!")
    print(f"  Succeeded: {total_success} folders")
    print(f"  Failed: {total_fail} folders")
    print(f"  Output directory: {output_base_dir}")
    print("="*60)


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
    from paths import DATA_ROOT
    ap = argparse.ArgumentParser(description="DICOM -> NIfTI batch conversion")
    ap.add_argument("--source-dir", default=os.path.join(DATA_ROOT, "dicom", "external"))
    ap.add_argument("--target-dir", default=os.path.join(DATA_ROOT, "nifti", "external"))
    args = ap.parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    
    print("="*60)
    print("DICOM → NIfTI batch conversion tool")
    print("="*60)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {target_dir}")
    print("="*60)
    
    convert_dicoms_to_nifti(source_dir, target_dir)
