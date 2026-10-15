alpha_min=1
alpha_max=130
k_min=0.05
k_max=0.3
resolution=1000
n_chunks=40

with open("parameters.txt", 'w') as file:
        for i in range(n_chunks):
            for j in range(n_chunks):
                file.writelines(f"{i} {j}\n")

job_str = f"""Executable = run.sh
arguments = {alpha_min} {alpha_max} {k_min} {k_max} {resolution} {n_chunks} $(chunk_alpha) $(chunk_k)

Error = job.$(Cluster).err
Log   = job.$(Cluster).log
Output = chain.$(Cluster).csv

request_memory = 128MB

+RequestFreeDisk = 10*1024
+MaxRuntime = 23*3600
Queue chunk_alpha,chunk_k from parameters.txt
"""

with open("job.jdf", "w") as file:
    file.write(job_str)
