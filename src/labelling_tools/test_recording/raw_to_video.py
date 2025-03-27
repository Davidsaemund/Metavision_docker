import argparse
import subprocess

"""
Python script to run source code from prophesee metavision sdk module found here :
https://docs.prophesee.ai/stable/samples/modules/core/file_to_video.html 


Usage:
python3 raw_to_video.py -n <name1> <name2> ... -i <num_runs>

Example:
python3 raw_to_video.py -n left right center -i 5
"""

def run_metavision_conversion(names, num_runs):
    for name in names:
        for i in range(1, num_runs + 1):  # Start the loop from 1
            input_file = f"{name}_{i}.raw"
            output_file = f"{name}_{i}.avi"
            slow_factor = "1.333333"
            acc_time = "25000"
            command = ["metavision_file_to_video", "-i", input_file, "-o", output_file, "-s", slow_factor, "-a", acc_time ]

            try:
                subprocess.run(command, check=True)
                print(f"Conversion {name}_{i} completed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred during conversion {name}_{i}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run metavision conversion for multiple input files.")
    parser.add_argument("-n", "--names", nargs="+", help="List of input file prefixes", required=True)
    parser.add_argument("-i", "--num_runs", type=int, help="Number of runs for each input file prefix", required=True)
    args = parser.parse_args()

    run_metavision_conversion(args.names, args.num_runs)
