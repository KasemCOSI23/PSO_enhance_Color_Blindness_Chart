import numpy as np
import matplotlib.pyplot as plt
import random
# import cv2
from CSL_2024_Python_codes import *


# 3. Calculate spectral reflectance image Rsample(x,y,λ)
Sample, Wavelength  = read_envi('capture\sample_scan_0009.hdr')
Darkref, _ = read_envi('capture\DARKREF_sample_scan_0009.hdr')
Whiteref, _ = read_envi('capture\WHITEREF_sample_scan_0009.hdr')

print(np.shape(Wavelength))
print(np.shape(Sample))
print(np.shape(Darkref))
print(np.shape(Whiteref))

mean_darkref = np.mean(Darkref, axis=0)
print(f"Shape of the mean_darkref: {np.shape(mean_darkref)}")
mean_whiteref = np.mean(Whiteref, axis=0)
print(f"Shape of the mean_whiteref: {np.shape(mean_whiteref)}")

replicated_darkref = np.tile(mean_darkref, (np.shape(Sample)[0], 1, 1))  
print(f"Shape of the replicated mean dark reference: {np.shape(replicated_darkref)}")
replicated_whiteref = np.tile(mean_whiteref, (np.shape(Sample)[0], 1, 1))  
print(f"Shape of the replicated mean white reference: {np.shape(replicated_whiteref)}")

# Calcuate Calculate spectral reflectance image
reflectance_image = (Sample - replicated_darkref) / (replicated_whiteref- replicated_darkref)

# 4. Limit wavelength range to 400 – 700 nm (remove: 700 – 1000 nm).
valid_indices = np.where((Wavelength >= 400) & (Wavelength <= 700))[0]

Wavelength_filtered = Wavelength[valid_indices]
reflectance_image_filtered = reflectance_image[:, :, valid_indices]

print(f"Filtered Wavelength shape: {np.shape(Wavelength_filtered)}")
print(f"Filtered Reflectance Image shape: {np.shape(reflectance_image_filtered)}")

# 5. Crop area-of-interest from the spectral image (color blindness test chart).
RGB_image = spim2rgb(reflectance_image_filtered, Wavelength_filtered, lsource='D65', clip_min=0, clip_max=1)
print(f'RGB_image_shape: {np.shape(RGB_image)}')

# roi = cv2.selectROI("Select Region", RGB_image)
# cv2.destroyAllWindows()

x1, y1 = 703, 2
x2 = 1403
y2 = 641

# print(f"  Top-left corner (x1, y1): ({x1}, {y1})")
# print(f"  Width: {width}")
# print(f"  Height: {height}")
# print(f"  Bottom-right corner (x2, y2): ({x2}, {y2})")

cropped_image = reflectance_image_filtered[y1:y2, x1:x2, :]

rgb_cropped_image = spim2rgb(cropped_image, Wavelength_filtered, lsource='D65', clip_min=0, clip_max=1)

plt.imshow(rgb_cropped_image)
plt.title("Cropped Spectral Image")
plt.show()

print(f"Shape of the cropped reflectance image: {cropped_image.shape}")

# 6. Select two reflectance spectra R1(λ) and R2(λ) from two different-colored points from the color blindness test chart (e.g., from red and green spots).
# fig, ax = plt.subplots()
# plt.imshow(rgb_cropped_image)
# plt.title('Select 2 points (patches) from the image')
# plt.axis('off')
# points = plt.ginput(2)  # Manually select 3 points
# plt.close()

# plt.figure()

# print(f"Points: {points}")

points = [(254.76623376623374, 336.7727272727273), (236.87012987012983, 240.40909090909093)]

plt.imshow(rgb_cropped_image)
plt.title("Cropped Spectral Image with Selected Points")
plt.axis('off')
# Unzip the points into separate x and y coordinates
x_points, y_points = zip(*points)

# Plot black 'X' markers at the specified points
plt.scatter(x_points, y_points, color='black', marker='x', s=100, label='Selected Points')

# Add a legend
plt.legend()

# Show the image with points
plt.show()

print(f"Points plotted: {points}")

for point in points:
    x, y = int(point[0]), int(point[1])

    reflectance = cropped_image[y, x, :]
    print(np.shape(reflectance))
    plt.plot(Wavelength_filtered, reflectance, label=f'Point ({x}, {y})')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.legend()
plt.show()


two_reflectance = []
for point in points:
    x, y = int(point[0]), int(point[1])
    reflectance = cropped_image[y, x, :]
    two_reflectance.append(reflectance)

# 7. Read the measured LED-spectra LLED, j(λ) , j=1,…,10, and interpolate them to the same wavelength range as the spectral image Rsample(x,y,λ)
led_spectra = []  # To store interpolated spectra

for i in range(1, 11):  # Loop for 10 LEDs (led1.txt, led2.txt, ..., led10.txt)
    filename = f'Hamamatsu_LEDs\led{i}.txt'  # Construct filename
    
    # Read the LED spectrum
    led_data = np.loadtxt(filename)
    wavelength_led = led_data[:, 0]
    intensity_led = led_data[:, 1]
    
    # Interpolate to match the wavelength range of the spectral image
    interpolated_intensity = np.interp(Wavelength_filtered, wavelength_led, intensity_led)
    
    # Store the interpolated spectrum
    led_spectra.append(interpolated_intensity)

led_spectra = np.array(led_spectra)  # Convert list to a numpy array
print(led_spectra.shape)

# Part 2: Preparation for Particle Swarm Optimization:

def rgb2gray(RGB_image):
    return np.dot(RGB_image[..., :3], [0.2125, 0.7154, 0.0721])

def deltaE(two_reflectance, emission_spectrum, Wavelength = Wavelength_filtered): 
    RGB1 = spim2rgb(two_reflectance[0], Wavelength, lsource=emission_spectrum, clip_min=np.nan, clip_max=np.nan)
    RGB2 = spim2rgb(two_reflectance[1], Wavelength, lsource=emission_spectrum, clip_min=np.nan, clip_max=np.nan)

    if np.isnan(RGB1).any() or np.isnan(RGB2).any():
        return np.nan
    XYZ1 = spim2XYZ(two_reflectance[0], Wavelength, lsource=emission_spectrum)
    Lab1 = XYZ2Lab(XYZ1, Wavelength, cie_illuminant=emission_spectrum)
    XYZ2 = spim2XYZ(two_reflectance[1], Wavelength, lsource=emission_spectrum)
    Lab2 = XYZ2Lab(XYZ2, Wavelength, cie_illuminant=emission_spectrum)

    delta_L = Lab1[..., 0] - Lab2[..., 0]
    delta_a = Lab1[..., 1] - Lab2[..., 1]
    delta_b = Lab1[..., 2] - Lab2[..., 2]
    
    return np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

def deltaRGB(two_reflectance, emission_spectrum, Wavelength = Wavelength_filtered):
    RGB1= spim2rgb(two_reflectance[0], Wavelength, lsource= emission_spectrum, clip_min=np.nan, clip_max=np.nan)
    RGB2= spim2rgb(two_reflectance[1], Wavelength, lsource= emission_spectrum, clip_min=np.nan, clip_max=np.nan)
    if np.isnan(RGB1).any() or np.isnan(RGB2).any():
        return np.nan
    return np.sqrt(np.sum((RGB1 - RGB2) ** 2))

def michelson_contrast(two_reflectance, emission_spectrum, Wavelength = Wavelength_filtered):
    RGB1= spim2rgb(two_reflectance[0], Wavelength, lsource= emission_spectrum, clip_min=np.nan, clip_max=np.nan)
    RGB2= spim2rgb(two_reflectance[1], Wavelength, lsource= emission_spectrum, clip_min=np.nan, clip_max=np.nan)  
    if np.isnan(RGB1).any() or np.isnan(RGB2).any():
        return np.nan 
    g1 = rgb2gray(RGB1)
    g2 = rgb2gray(RGB2)   
    max_val = max(g1, g2)
    min_val = min(g1, g2)
    contrast = (max_val - min_val) / (max_val + min_val)
    return contrast

def COST(two_reflectance, emission_spectrum, method, Wavelength = Wavelength_filtered):
    if method == 'deltaE':
        return deltaE(two_reflectance, emission_spectrum, Wavelength = Wavelength_filtered)
    elif method == 'deltaRGB':
        return deltaRGB(two_reflectance, emission_spectrum, Wavelength = Wavelength_filtered)
    elif method == 'contrast':
        return michelson_contrast(two_reflectance, emission_spectrum, Wavelength = Wavelength_filtered)
    else:
        raise ValueError("Unknown cost function method")
    


    
def PSO_algo(n_particles, n_iterations, two_reflectance, emission_spectrum, Wavelength_filtered, minimize=True):
    
    # Step 1: Set PSO parameters
    C0, C1, C2 = 0.95, 2, 2  # Inertia, personal, and global components
    N = n_particles  # Number of particles
    
    # Step 2: Initialize particle positions and velocities
    particles_position = np.random.uniform(0, 1, (n_particles, 10))  # 10 LED intensities
    particles_velocity = np.random.uniform(-1, 1, (n_particles, 10))  # Velocities
    
    
    # Initialize global and local bests
    if minimize:
        global_best_value = float('inf')
    else:
        global_best_value = -float('inf')
    
    global_best_position = np.random.uniform(0, 1, 10)
    local_best_position = np.copy(particles_position)

    L_init = np.array([np.sum(p[:, None] * emission_spectrum, axis=0) for p in particles_position])
    print(L_init.shape)

    local_best_value = np.full(n_particles, np.inf)  # Initialize with inf for minimization
    for i, L in enumerate(L_init):
        cost_value = COST(two_reflectance, L, method='deltaRGB', Wavelength = Wavelength_filtered)
        if np.isnan(cost_value):
            cost_value = np.inf if minimize else -np.inf  # Set to a high value to ignore it
        local_best_value[i] = cost_value  # Always set a value here 
    print(f"Shape of local_best_value: {local_best_value.shape}")


    # Step 3: PSO main loop
    for iteration in range(n_iterations):
        for i in range(n_particles):
            # Step 1: Update velocities and positions
            r1, r2 = np.random.random(), np.random.random()
            particles_velocity[i] = (
                C0 * particles_velocity[i]
                + C1 * r1 * (local_best_position[i] - particles_position[i])
                + C2 * r2 * (global_best_position - particles_position[i])
            )
            particles_position[i] += particles_velocity[i]

            # Step 2: Enforce constraints (non-negative and normalized)
            particles_position[i] = np.maximum(particles_position[i], 0)  # Non-negative
            particles_position[i] /= np.sum(particles_position[i])  # Sum of weights = 1

            # Step 3: Calculate emission spectrum L_i(λ) for this particle
            Li = np.sum(particles_position[i][:, None] * emission_spectrum, axis=0)


            cost_value = COST(two_reflectance, Li, method='deltaRGB', Wavelength = Wavelength_filtered)  # You can switch method ('deltaE', 'deltaRGB', 'contrast')

            if np.isnan(cost_value):
                continue  # Skip this particle if the cost is NaN

            # Step 5: Update local best if needed
            if minimize and cost_value < local_best_value[i]:
                local_best_value[i] = cost_value
                local_best_position[i] = np.copy(particles_position[i])
            elif not minimize and cost_value > local_best_value[i]:
                local_best_value[i] = cost_value
                local_best_position[i] = np.copy(particles_position[i])

            # Step 6: Update global best if needed
            if minimize and cost_value < global_best_value:
                global_best_value = cost_value
                global_best_position = np.copy(particles_position[i])
            elif not minimize and cost_value > global_best_value:
                global_best_value = cost_value
                global_best_position = np.copy(particles_position[i])

        # To this:
        if isinstance(global_best_value, np.ndarray):
            global_best_value = global_best_value.item()  # Convert to scalar if it's an array

        print(f"Iteration {iteration + 1}/{n_iterations}, Global Best Value: {global_best_value:.6f}")

    # Return the best particle position found
    print(f"Optimal solution: {global_best_position}, with cost: {global_best_value:.6f}")
    return global_best_position

# # Example parameters
n_particles = 5000  # Large number of particles
n_iterations = 50  # Number of iterations

emission_spectrum = led_spectra

# Run the PSO algorithm
global_best_position = PSO_algo(n_particles, n_iterations, two_reflectance, emission_spectrum, Wavelength_filtered, minimize=True)

# Calculate the optimized illuminant spectrum L_optim(λ)
L_optim = np.sum(global_best_position[:, None] * emission_spectrum, axis=0)

def plot_spectra(L_optim, R1, R2, wavelengths):
    # Creating subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # Plot L_optim
    axs[0].plot(wavelengths, L_optim, label="L_optim(λ)", color='blue', linewidth=2)
    axs[0].set_title("Optimal Illuminant")
    axs[0].set_xlabel("Wavelength (nm)")
    axs[0].set_ylabel("Illuminant Value")
    axs[0].legend()
    axs[0].grid(True)

    # Plot R1
    axs[1].plot(wavelengths, R1, label="R1(λ)", color='red', linestyle='dashed')
    axs[1].set_title("Sample 1 Reflectance")
    axs[1].set_xlabel("Wavelength (nm)")
    axs[1].set_ylabel("Reflectance")
    axs[1].legend()
    axs[1].grid(True)

    # Plot R2
    axs[2].plot(wavelengths, R2, label="R2(λ)", color='green', linestyle='dotted')
    axs[2].set_title("Sample 2 Reflectance")
    axs[2].set_xlabel("Wavelength (nm)")
    axs[2].set_ylabel("Reflectance")
    axs[2].legend()
    axs[2].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()

plot_spectra(L_optim, two_reflectance[0], two_reflectance[1], Wavelength_filtered)

Minimize_E_image = spim2rgb(cropped_image, Wavelength_filtered, lsource=L_optim, clip_min=0, clip_max=1)
plt.imshow(Minimize_E_image)
plt.title("Minimize_deltaRGB_image")
plt.show()


data_to_save = np.column_stack((Wavelength_filtered, L_optim))

output_filename = 'Minimize_deltaRGB_optim_L.txt'
np.savetxt(output_filename, data_to_save, fmt='%.6f', delimiter=' ')

print(f"Data saved to {output_filename}.")