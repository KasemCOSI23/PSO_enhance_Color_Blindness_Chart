# PSO_enhance_Color_Blindness_Chart
This project focuses on designing optimal illumination spectra using Particle Swarm Optimiza-tion (PSO) to minimize or maximize differences between selected points in a color blindness test chart.

# Data collection
We used Specim V10E camera to capture and process spectral data (400–1000 nm)

## Installation

### Requirements
- Python 3.8 or higher
- The following Python libraries:
  - `numpy`
  - `matplotlib`
  - `opencv-python`
  
### Setup Instructions
1. **Clone the repository** to your local machine:
   ```bash
   https://github.com/KasemCOSI23/PSO_enhance_Color_Blindness_Chart.git
   ```
2. **Runnung Practical_work.py file in VScode or other software you want**
   
   However, you need your own Sample, Dark reference and White reference data:
   ```bash
   Sample, Wavelength  = read_envi('your own Sample path')
   Darkref, _ = read_envi('your own Darkref path')
   Whiteref, _ = read_envi('your own Whiteref path')
   ```
### Result
#### You can see some examples of result in Result folder.

### What does this Practical_work.py file do
1. **Calculate spectral reflectance image** 
2. **Limit wavelength range to 400 – 700 nm**
3. **Crop area-of-interest from the spectral image**
4. **Select two reflectance spectra from two different-colored points**
5. **Read the measured LED-spectra LLED, j(λ) , j=1,…,10, and interpolate them to the same wavelength range**
6. **Setup and Run Particle Swarm Optimization**
7. **Calculate the optimized illuminant spectrum**
   

## Acknowledgements
This project is part of the Color Science Laboratory course 2024 at the University of Eastern Finland
Kasem and Yeshi especially thanks Dr. Pauli Fält for his support in developing this project.
