import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
df = pd.read_csv('timesurface_classification_inference.csv')

# Plot the classification scores over frames
plt.figure(figsize=(10, 6))
plt.plot(df['frame'], df['background'], label='Background')
plt.plot(df['frame'], df['left'], label='Left')
plt.plot(df['frame'], df['center'], label='Center')
plt.plot(df['frame'], df['right'], label='Right')
plt.xlabel('Frame')
plt.ylabel('Score')
plt.title('Classification Scores for Timesurface')
plt.legend()
plt.show()
