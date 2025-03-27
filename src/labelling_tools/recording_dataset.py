import subprocess
import pyautogui
import time

def record_data(output_name):
    # Start the metavision_viewer process /// -b testing_1.bias --output-camera-config settings_testing_1.json 
    command = f"metavision_viewer -o {output_name}.raw"
    process = subprocess.Popen(command, shell=True)
    
    # Wait for 2 seconds
    time.sleep(7)
    
    # Simulate pressing space to start recording
    pyautogui.press('space')
    
    # Wait for 3 seconds
    time.sleep(40)
    
    # Simulate pressing space to stop recording
    pyautogui.press('space')
    pyautogui.press('q')
    # Wait for the process to finish
    process.wait()

# Run the script in a loop 5 times with different output names
for i in range(1, 31):
    record_data(f"test_recording_{i}")
