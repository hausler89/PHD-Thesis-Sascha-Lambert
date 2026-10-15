"""Executable = run.sh
arguments = {alpha1_min} {alpha1_max} {alpha2_min} {alpha2_max} {resolution} {n_chunks} $(chunk_alpha1) $(chunk_alpha2)

Error = job.$(Cluster).err
Log   = job.$(Cluster).log
Output = chain.$(Cluster).csv

request_memory = 128MB

+RequestFreeDisk = 10*1024
+MaxRuntime = 23*3600
Queue chunk_alpha1,chunk_alpha2 from parameters.txt
"""

with open("job.jdf", "w") as file:
    file.write(job_str)
