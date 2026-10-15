# ABP Posterior Survey 1

Here, we study a dataset of 10 ABPs, scattered at a WCA wall. The
survey is divided into multiple steps.

The torque model is Merle Duchene's parametrization, where ALPHA
denotes the prefactor to the torque and k the scaling of the
angle dependency.

## Global data

In the "GLOBAL" directory, we collect data and code that is used
throughout the survey. For example, the analyzed data lives in
that directory, as well as torque definitions and ground truths.

## Step 1 - Data generation

We start with the generation of the data. This is done in the
according directory. We use a fixed seed of 12345 for NumPy.

## Step 2 - Coarse overview over the Posterior

A coarse overview over the posterior using a Kalman filter.

## Step 3 - High Resolution Sampling on the HPC cluster

This step is executed on the HPC cluster. We subdivide the area
of interest into chunks and process them individually - and in
parallel.

After all jobs are finished, we consolidate the data and process
it locally again.

