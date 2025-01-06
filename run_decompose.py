import numpy as np
import matplotlib.pyplot as plt
from astropy.io.votable import parse
import glob

import sys,os
import Gaussian_fitting as Gf
#sys.path.append('../Absorption_fitting')
import spectra_decomposing as sd
import read_GASKAP_data as rd
import time
from multiprocessing import Process, Queue

datapathbase='/d/bip5/hchen'


def delete_csv_txt_files(directory,file='.csv'):
    """
    Deletes all .csv and .txt files in the specified directory.

    Args:
        directory (str): Path to the directory.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Construct the full file path
        file_path = os.path.join(directory, filename)

        # Check if it's a file and has .csv or .txt extension
        if os.path.isfile(file_path) and (filename.endswith(file)):
            try:
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def process_iteration(j, queue):
    """Function to handle the processing for each j."""
    try:
        print(f"Processing j={j} (PID: {os.getpid()})")
        x, y, yerr, xemi, yemi, yemi_err, yemi_selferr = rd.get_emi_abs_data(
            nlist[j], nabs[j], 1, 1, mode='LMC', v_move=0.
        )

        fig = plt.figure(figsize=(12, 8), dpi=300)
        ax1_first = plt.subplot2grid((8, 2), (0, 0), rowspan=3, fig=fig)

        ax = [ax1_first,
              plt.subplot2grid((8, 2), (3, 0), rowspan=1, fig=fig, sharex=ax1_first),
              plt.subplot2grid((8, 2), (4, 0), rowspan=3, fig=fig, sharex=ax1_first),
              plt.subplot2grid((8, 2), (7, 0), fig=fig, sharex=ax1_first)]

        sd.fit_and_plot(x, y, yerr, xemi, yemi, yemi_err, ax, name=nlist[j], savetxt=True,
                        peak_abs=[], peak_emi=[], Tsmin=3.77,
                        fit_mode='BIC', v_sh=4)

        fig.savefig(datapathbase + '/output_data/LMC_fitting/plots/%s.png' % (nlist[j]), dpi=300, bbox_inches='tight')
        sys.stdout.flush()
        queue.put(None)  # Success
    except Exception as e:
        queue.put(str(e))  # Return the exception as a string


def process_wrapper(j, timeout):
    """Wrapper to enforce timeout and handle the process."""
    queue = Queue()
    process = Process(target=process_iteration, args=(j, queue))
    process.start()
    process.join(timeout=timeout)

    if process.is_alive():
        print(f"Timeout for j={j}. Terminating process.")
        process.terminate()
        process.join()
        return j, "Timeout"
    else:
        result = queue.get()  # Get the result from the queue
        if result is None:
            return j, None  # Success
        else:
            return j, result  # Error


def parallel_processing(j_values, max_concurrent_processes=4, timeout=20):
    """Run multiple `j` processes in parallel with timeout enforcement."""
    results = []
    active_processes = {}
    completed_processes = 0

    while completed_processes < len(j_values):
        # Start new processes if we haven't reached the limit
        while len(active_processes) < max_concurrent_processes and j_values:
            j = j_values.pop(0)  # Get the next `j`
            p = Process(target=process_wrapper, args=(j, timeout))
            p.start()
            active_processes[j] = p

        # Check for completed processes
        for j, process in list(active_processes.items()):
            if not process.is_alive():
                process.join()
                result = process_wrapper(j, timeout=timeout)
                results.append(result)
                active_processes.pop(j)
                completed_processes += 1

                # Log timeouts and errors
                if result[1] == "Timeout":
                    f.write(f"{nlist[j]}\n")  # Record the long-running iteration
                    f.flush()  # Ensure immediate write to disk
                    os.fsync(f.fileno())  # Force file system sync
                    print(f"Timeout for j={j}. Logging to long_runs.txt.")
                elif result[1] is not None:
                    print(f"Error for j={j}: {result[1]}")

        time.sleep(0.1)  # Avoid busy-waiting

    return results



if __name__ == "__main__":
    # Example usage
    directory_to_check = datapathbase+"/output_data/LMC_fitting"
    #delete_csv_txt_files(directory_to_check,file='.csv')
    #delete_csv_txt_files(directory_to_check,file='.txt')
    #delete_csv_txt_files(directory_to_check+'/plots',file='.png')
    #idx = int(sys.argv[1])
    #print sources observed twice
    nemi=np.sort(glob.glob(datapathbase+'/LMC_GASKAP_emi_all_abs/*.txt'))
    nabs_n=glob.glob(datapathbase+'/LMC_abs_new/sb*')
    nabs=[None] * len(nemi)
    for i, ni in enumerate(nemi):
        na_=ni[-18:-4]
        for nj in nabs_n:
            _=glob.glob(nj+'/spectra_abs/*')
            _all=[filename[-23:-9] for filename in _]
            if na_ in _all:
                if nabs[i] is not None:
                    #print(i,nj,na_)
                    break  # Exit the loop once a match is found for this `na_`
                else:
                    nabs[i] = f'{nj}/spectra_abs/{na_}_spec.vot'  # Record `nj` only once for this `na_`
                    
    nemi,nabs=np.array(nemi),np.array(nabs)  
    nlist=[filename[-18:-4] for filename in nemi]
    #0-18:original
    #19-84:not change Gaussian fitting
    # Define a timeout handler


     # Prepare the list of `j` values to process
    j_values = list(range(96, 222))

    # Timeout and number of concurrent processes
    max_concurrent_processes = 20  # Number of parallel processes
    timeout = 30*60  # Timeout for each task in seconds

    # Open the file to log long-running iterations
    with open(datapathbase + '/output_data/LMC_fitting/long_runs.txt', 'a') as f:
        results = parallel_processing(j_values, max_concurrent_processes=max_concurrent_processes, timeout=timeout)

    
            #print(j,nlist[j],end='\r', flush=True)
            #nohup /bin/python3 run_decompose.py > output.log 2>&1 &
            #ps aux | grep run_decompose.py
                    