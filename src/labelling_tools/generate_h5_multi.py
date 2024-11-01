import argparse
import subprocess
import numpy as np
from metavision_ml.preprocessing import get_preprocess_function_names, get_preprocess_dict
from metavision_ml.preprocessing.hdf5 import generate_hdf5
"""
Python script to run source code from prophesee metavision sdk module found here :
https://docs.prophesee.ai/stable/samples/modules/core/file_to_video.html 

python3 generate_hdf5.py EVK_4_LCR_dataset/test/right.raw -o EVK_4_LCR_dataset/test --delta-t 10000 --preprocess histo_quantized --neg_bit_len_quantized 4 --total_bit_len_quantized 8 --normalization_quantized --num-workers
 32 --height_width 360 640
Usage:
python3 generate_h5_multi.py -n <name1> <name2> ... -i <num_runs>

Example:
python3 raw_to_video.py -n left right center -i 5
"""

def run_metavision_conversion(data_folder, names, num_runs):
    for name in names:
        for i in range(1, num_runs + 1):  # Start the loop from 1
            if num_runs == 1:
                input_file = f"{data_folder}/{name}.raw"
            else:
                input_file = f"{data_folder}/{name}_{i}.raw"
            
            #command = f"{input_file}, -o {data_folder}, --delta-t 10000, --preprocess histo_quantized, --neg_bit_len_quantized 4, --total_bit_len_quantized 8, --normalization_quantized, --num-workers 32, --height_width 360 640"
            preprocess_kwargs = get_preprocess_dict("histo_quantized")['kwargs']
            preprocess_kwargs.update({"negative_bit_length": 4,
                            "total_bit_length": 8,
                            "normalization": True,
                            "preprocess_dtype": np.float32})
            try:
                generate_hdf5(paths=input_file,
                            output_folder=data_folder,
                            preprocess="histo_quantized",
                            delta_t=25000,
                            n_processes=32, 
                            height=360,
                            width=640,
                            preprocess_kwargs=preprocess_kwargs)
                
                print(f"Conversion {name}_{i} completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred during conversion {name}_{i}: {e}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metavision conversion for multiple input files.")
    parser.add_argument("-d", "--data_folder", type=str, help="Name of dataset folder for converting", required=True)
    parser.add_argument("-n", "--names", nargs="+", help="List of input file prefixes", required=True)
    parser.add_argument("-i", "--num_runs", type=int, help="Number of runs for each input file prefix", required=True)
    args = parser.parse_args()

    run_metavision_conversion(args.data_folder, args.names, args.num_runs)
