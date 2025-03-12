import numpy as np
import matplotlib.pyplot as plt
from astropy.io.votable import parse
import glob
import sys
import os
from radiative_transfer import Gf,sd
import read_GASKAP_data as rd
import time
from multiprocessing import Process, Queue

datapathbase = '/d/bip5/hchen'


def delete_csv_txt_files(directory, file='.csv'):
    """
    Deletes all .csv and .txt files in the specified directory.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith(file):
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


def process_iteration(j, queue):
    """Function to handle the processing for each j."""
    try:
        print(f"Processing j={j} (PID: {os.getpid()})")
        spectra= rd.get_emi_abs_data(
            nlist[j], nabs[j], 1, 1, mode='LMC', v_move=0.,save_saturated=True
        )
        x,y,yerr,xemi,yemi,yemi_err=spectra['velocity_tau'],spectra['1_e_tau'],spectra['1_e_tau_err'],spectra['velocity_TB'],spectra['TB'],spectra['TB_err']


        fig = plt.figure(figsize=(12, 8), dpi=300)
        ax1_first = plt.subplot2grid((8, 2), (0, 0), rowspan=3, fig=fig)

        ax = [ax1_first,
              plt.subplot2grid((8, 2), (3, 0), rowspan=1, fig=fig, sharex=ax1_first),
              plt.subplot2grid((8, 2), (4, 0), rowspan=3, fig=fig, sharex=ax1_first),
              plt.subplot2grid((8, 2), (7, 0), fig=fig, sharex=ax1_first)]

        #sd.fit_and_plot(x, y, yerr, xemi, yemi, yemi_err, ax, name=nlist[j], savetxt=True,
        #                peak_abs=[], peak_emi=[], Tsmin=3.77,
        #                fit_mode='BIC', v_sh=4)
        
        tsyn,_=rd.read_synchro_emi(nlist[j],radius=60)
        spec_fit=sd(x,y,yerr,xemi,yemi,yemi_err)
        spec_fit.name=nlist[j]
        spec_fit.v_shift=4.
        spec_fit.peak_abs=[]
        spec_fit.peak_emi=[]
        spec_fit.Tsmin=2.73+tsyn
        spec_fit.Tsky=2.73+tsyn
        spec_fit.savecsv=True
        spec_fit.fit_mode='BIC'
        spec_fit.ax=ax
        spec_fit.datapath=datapathbase + '/output_data/LMC_fitting/'
        spec_fit.fit_and_plot()
        fig.savefig(datapathbase + '/output_data/LMC_fitting/plots/%s.png' % (nlist[j]), dpi=300, bbox_inches='tight')
        sys.stdout.flush()
        queue.put(None)  # Success
    except Exception as e:
        queue.put(str(e))  # Return the exception as a string

def parallel_processing(j_values, max_concurrent_processes=1, timeout=20):
    """Run multiple `j` processes in parallel with timeout enforcement."""
    results = []
    active_processes = {}  # Track active processes

    while j_values or active_processes:  # Continue until no tasks or active processes
        # Start new processes if we haven't reached the limit
        while len(active_processes) < max_concurrent_processes and j_values:
            j = j_values.pop(0)  # Get the next `j`
            queue = Queue()  # Create a queue for communication
            process = Process(target=process_iteration, args=(j, queue))
            process.start()
            active_processes[j] = (process, queue, time.time())  # Track the process, queue, and start time
            print(f"Started process for j={j}.")

        # Check for timeout or completed processes
        for j, (process, queue, start_time) in list(active_processes.items()):
            elapsed_time = time.time() - start_time

            # Handle timeout
            if elapsed_time > timeout:
                print(f"Timeout for j={j}. Terminating process.")
                process.terminate()  # Forcefully terminate the process
                process.join()  # Wait for cleanup
                results.append((j, "Timeout"))  # Log timeout
                active_processes.pop(j)  # Remove from active processes

            # Handle completed process
            elif not process.is_alive():
                process.join()  # Ensure the process is properly cleaned up
                result = queue.get()  # Get the result from the queue
                if result is None:
                    results.append((j, None))  # Success
                else:
                    results.append((j, result))  # Error
                active_processes.pop(j)  # Remove from active processes

        # Allow other processes to run before checking again
        time.sleep(0.1)

    return results






if __name__ == "__main__":
    directory_to_check = datapathbase + "/output_data/LMC_fitting"
    delete_csv_txt_files(directory_to_check,file='.csv')
    #delete_csv_txt_files(directory_to_check,file='.txt')
    delete_csv_txt_files(directory_to_check+'/plots',file='.png')

    nemi=np.sort(glob.glob(datapathbase+'/LMC_GASKAP_emi_all_abs/*.txt'))
    nabs_n=glob.glob(datapathbase+'/LMC_abs_new/abs_v1.0/sb*')
    nabs=[None] * len(nemi)
    for i, ni in enumerate(nemi):
        na_=ni[-18:-4]
        for nj in nabs_n:
            _=glob.glob(nj+'/averaged/spectra/*_spec.vot')
            _all=[filename[-23:-9] for filename in _]
            if na_ in _all:
                if nabs[i] is not None:
                    print(i,nj,na_)
                    break  # Exit the loop once a match is found for this `na_`
                else:
                    nabs[i] = f'{nj}/averaged/spectra/{na_}_spec.vot'  # Record `nj` only once for this `na_`
                    
    nemi,nabs=np.array(nemi),np.array(nabs)  
    nlist=[filename[-18:-4] for filename in nemi]
    nlist=np.array(nlist)        

    
    a= glob.glob(datapathbase + '/output_data/LMC_fitting/plots/*.png')
    nfinish=np.array([path[-18:-4] for path in a])
    # Prepare the list of `j` values to process
    j_values = np.where(~np.isin(nlist, nfinish))[0]
    j_values=j_values.tolist()
    #j_values=j_values[0:2]
    #print(j_values)

    # Timeout and number of concurrent processes
    max_concurrent_processes = 29  # Number of parallel processes
    timeout = 25*60*60  # Timeout for each task in seconds

    # Open the file to log long-running iterations
    with open(datapathbase + '/output_data/LMC_fitting/long_runs.txt', 'a') as f:
        results = parallel_processing(j_values, max_concurrent_processes=max_concurrent_processes, timeout=timeout)
        print(results)

        for j, result in results:
            if result == "Timeout":
                print(f"Timeout for j={j}. Logging to long_runs.txt.")
                f.write(f"{nlist[j]}\n")
                f.flush()
            elif result is not None:
                print(f"Error for j={j}: {result}")
                
            else:
                print(f"Successfully processed j={j}.")


#nohup /bin/python3 run_decompose.py > output.log 2>&1 &
            #ps aux | grep run_decompose.py
            #ls -1 *.png 2>/dev/null | wc -l
            #pkill -f run_decompose.py