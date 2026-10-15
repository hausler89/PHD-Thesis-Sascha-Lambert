alpha1_min=1
alpha1_max=500
alpha2_min=1
alpha2_max=500
resolution=2000
n_chunks=40

with open("parameters.txt", 'w') as file:
        for i in range(n_chunks):
            for j in range(n_chunks):
                file.writelines(f"{i} {j}\n")

job_str = f"""Executable = run.sh
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
