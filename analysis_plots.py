import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D, NoOverlapError
from scipy.stats import gaussian_kde, zscore
from scipy.interpolate import griddata



# Sample DataFrames
# Assuming 'CNM' and 'EBV_origin' are already defined with 'Ra', 'Dec', 'Ra1', 'Dec1'
# CNM = pd.DataFrame({'Ra': [10, 20], 'Dec': [30, 40]})
# EBV_origin = pd.DataFrame({'Ra1': [10.1, 20.1], 'Dec1': [30.1, 40.1], 'Additional_Info': ['info1', 'info2']})


def add_EBV(CNM, EBV_origin):
    # Convert CNM RA and Dec to SkyCoord object
    r = CNM['ra'].to_numpy().astype(float)
    d = CNM['dec'].to_numpy().astype(float)
    cnm_coords = SkyCoord(ra=r * u.degree, dec=d * u.degree)

    # Convert EBV_origin RA1 and Dec1 to SkyCoord object
    ebv_coords = SkyCoord(ra=EBV_origin['RA'].values * u.degree, dec=EBV_origin['DEC'].values * u.degree)

    # Calculate the matrix of separations between each pair of CNM and EBV_origin points
    separation_matrix = cnm_coords[:, np.newaxis].separation(ebv_coords).arcminute

    # Initialize new columns in CNM DataFrame
    CNM['ra_ebv'] = np.nan
    CNM['dec_ebv'] = np.nan
    CNM['distance_arcmin'] = np.nan

    # Iterating through each CNM point to find matches within 1 arcmin
    for i in range(len(cnm_coords)):
        close_indices = np.where(separation_matrix[i] < 10)[0]
        if close_indices.size > 0:
            # Calculating mean RA and Dec from EBV_origin based on close matches
            mean_RA = EBV_origin.iloc[close_indices]['RA'].mean()
            mean_DEC = EBV_origin.iloc[close_indices]['DEC'].mean()
            mean_distance = separation_matrix[i][close_indices].mean()

            # Storing results in CNM
            CNM.at[i, 'ra_ebv'] = mean_RA
            CNM.at[i, 'dec_ebv'] = mean_DEC
            CNM.at[i, 'distance_arcmin'] = mean_distance

    # Optionally, merge additional data from EBV_origin if needed
    # Example: merging a specific column like 'EBV_value' based on closest mean values
    # This step would require further details on what needs to be merged and how.

    return CNM




def cdf(_data, bins=30):
    data = _data[~np.isnan(_data)] 
    hist, bin_edges = np.histogram(data, bins=bins, density=True)
    cdf = np.cumsum(hist * np.diff(bin_edges))
    cdf = np.insert(cdf, 0, 0)  # Insert 0 at the beginning of the cdf array

    # Adjust the bin_edges to include the start of the first bin
    gap = (bin_edges[1] - bin_edges[0]) / 2
    adjusted_bin_edges = np.insert(bin_edges[:-1]+gap, 0, bin_edges[0]+gap)

    return adjusted_bin_edges, cdf



def dis_close_shell(ra_list, dec_list,r_shell,d_shell):
    """
    Calculate the angular distance matrix for lists of RA and Dec.

    Parameters:
    ra_list : list or array of Right Ascension in degrees
    dec_list : list or array of Declination in degrees

    Returns:
    dist_matrix : A 2D numpy array containing angular distances in degrees
                  between each pair of coordinates.
    """
    # Create an array of SkyCoord objects from the RA and Dec lists
    coords1 = SkyCoord(ra=ra_list*u.degree, dec=dec_list*u.degree, frame='icrs')
    coords2 = SkyCoord(ra=r_shell*u.degree, dec=d_shell*u.degree, frame='icrs')
    
    # Initialize a matrix to hold the distances
    dist_matrix =[]
    for i in range(len(ra_list)):
        dm=100000
        for j in range(len(r_shell)):
            _d=coords1[i].separation(coords2[j]).degree
            if _d<dm:
                dm=_d
        dist_matrix.append(dm)

    return np.array(dist_matrix)*60



def plot_binned_data(x, y, num_bins=10, c='magenta'):
    """
    Bins the x data and calculates mean and standard deviation of corresponding y values per bin.
    Plots the mean y value per bin with error bars representing the standard deviation.

    Parameters:
    x (array-like): The x data.
    y (array-like): The y data corresponding to x data.
    num_bins (int): Number of bins to divide the x data into.

    Returns:
    None
    """
    # Create bins for the x data
    bins = np.linspace(min(x), max(x), num_bins+1)
    bin_indices = np.digitize(x, bins) - 1  # Bin indices for each x

    x_bin_centers = []
    y_bin_means = []
    y_bin_stds = []

    # Calculate mean and std of y values in each bin
    for i in range(num_bins):
        indices = bin_indices == i
        if np.sum(indices) >= 6:  # Check if the bin contains at least 3 samples
            y_bin_means.append(np.nanmean(y[indices]))
            y_bin_stds.append(np.nanstd(y[indices]))
            x_bin_centers.append((bins[i] + bins[i+1]) / 2)

    # Convert lists to numpy arrays for plotting
    x_bin_centers = np.array(x_bin_centers)
    y_bin_means = np.array(y_bin_means)
    y_bin_stds = np.array(y_bin_stds)

    # Plotting the results
    plt.errorbar(x_bin_centers, y_bin_means, yerr=y_bin_stds, fmt='o-', ecolor=c, capsize=2, markersize=2,color=c,linewidth=1)


# Function to calculate mean and std dev within a radius in arcseconds
def add_mask_mean(ra, dec, radius_arcsec, filename):
    with fits.open(filename) as hdul:
        image_data = hdul[0].data
        wcs = WCS(hdul[0].header)
    # Convert from RA, Dec to pixel coordinates
    ra=np.array(ra).astype(float)
    dec=np.array(dec).astype(float)
    x, y = wcs.wcs_world2pix(ra, dec, 0)

    # Check if the coordinates are within the image bounds
    if (x < 0 or y < 0 or x >= image_data.shape[1] or y >= image_data.shape[0]):
        return 0, 0  # Return zeros if outside bounds

    try:
        # Create a cutout around the position
        size = (radius_arcsec / 3600) / np.abs(wcs.wcs.cdelt[0]) * 2  # Diameter in pixels
        cutout = Cutout2D(image_data, (x, y), size, wcs=wcs, mode='partial', fill_value=np.nan)

        # Calculate the mean and std deviation, ignoring NaNs
        values = cutout.data
        values = values[~np.isnan(values)]  # Remove NaN values

        if values.size == 0:
            return 0, 0  # Return zeros if all are NaN
        else:
            return np.nanmean(values), np.nanstd(values)
    except NoOverlapError:
        return 0, 0  # Return zeros if no overlap with the image


def find_moment1(Ra,Dec,file_name,R_out=4,R_in=0):
    mom=[]
    
    fit = fits.open(file_name)
    a=WCS(fit[0].header)
    data=fit[1].data
        
    for i in range(len(Ra)):
        ra=Ra[i]
        dec=Dec[i]
        pix=a.all_world2pix([[ra,dec]], 1)
        #pix=a.all_world2pix([[ra,dec,190000]], 1)
        p1_=round(pix[0,1])
        p2_=round(pix[0,0])
        
        indices = np.indices(data.shape)
        distances = np.sqrt((indices[0] - p1_)**2 + (indices[1] - p2_)**2)
        outer_mask = distances <= R_out
        inner_mask = distances < R_in
        ring_mask = np.logical_xor(outer_mask, inner_mask)
        p=np.where(ring_mask)
        ye=data[ring_mask]
        ye=np.nanmean(ye)
        mom.append(ye)
    return np.array(mom)







def plot_density_contours(x, y, ax=None, threshold=3,c='r'):
    """
    Plots density contours excluding outliers based on Z-score.
    
    Parameters:
    x, y (array-like): Data points.
    ax (matplotlib.axes.Axes, optional): Axes object to draw the plot.
    threshold (float): Z-score threshold to identify outliers.
    """
    # Compute Z-scores for x and y
    zx = zscore(x)
    zy = zscore(y)

    # Filter out outliers
    mask = (np.abs(zx) < threshold) & (np.abs(zy) < threshold)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Continue as usual with density calculation
    xy = np.vstack([x_filtered, y_filtered])
    z = gaussian_kde(xy)(xy)

    if ax is None:
        fig, ax = plt.subplots()

    #scatter = ax.scatter(x_filtered, y_filtered, c=z, s=50, cmap='viridis')
    #plt.colorbar(scatter, ax=ax, label='Density')

    grid_x, grid_y = np.mgrid[min(x_filtered):max(x_filtered):100j, min(y_filtered):max(y_filtered):100j]
    grid_z = griddata((x_filtered, y_filtered), z, (grid_x, grid_y), method='cubic')
    ax.contour(grid_x, grid_y, grid_z, levels=4, linewidths=1.5, colors=c)


def plot_cdf_with_bins(x, y, bin_edges):
    """
    Plots the CDF for the y-values within manually assigned bins of x.

    Parameters:
    x (array-like): 1D array of x-values.
    y (array-like): 1D array of y-values.
    bin_edges (list): List of bin edges for dividing x-values.
    """
    
    # Ensure the bin edges are sorted
    bin_edges = np.sort(bin_edges)
    line_styles = ['-', '--', '-.', ':']
    
    # Loop through each pair of bin edges to create CDFs for each bin
    for i in range(len(bin_edges) - 1):
        # Create a mask for the current bin
        mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        # Select y-values that fall into the current bin
        bin_y = y[mask]

        # Sort the y-values and calculate the CDF
        sorted_y = np.sort(bin_y)
        cdf = np.arange(1, len(sorted_y) + 1) / float(len(sorted_y))

        # Plot the CDF for the current bin
        if i== len(bin_edges) - 2:
            plt.step(sorted_y, cdf, label=f'distance range (kpc): >{bin_edges[i]}',linewidth=2,)
        else:
            plt.step(sorted_y, cdf, label=f'distance range (kpc): [{bin_edges[i]}, {bin_edges[i + 1]})',linewidth=2,)



