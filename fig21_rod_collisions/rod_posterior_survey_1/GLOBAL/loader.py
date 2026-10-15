import os.path
import numpy as np
import os
import itertools
from shutil import rmtree
import subprocess




class ABPsim:
    def __init__(self):
        __init__(self, "./")

    def __init__(self, folder):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(module_dir, folder)
        print("Loading "+folder)
        self.folder = data_dir
        self.check_config_file()
        self.discover_times()

    def check_config_file(self):
        if not os.path.isfile(self.folder + "abp.conf"):
            raise Exception("Config file not found in folder " + self.folder)

        file = open(self.folder + "abp.conf", 'r')
        Lines = file.readlines()

        for line in Lines:
            if '=' in line and ';' not in line:
                try:
                    exec("self." + line.rstrip('\n'))
                except:
                    pass

    def discover_times(self):
        self.subfolders = [f for f in os.scandir(self.folder) if f.is_dir() & ('pycache' not in str(f))]

        self.time = [float(f.name.split('_')[0]) for f in self.subfolders]
        self.time_step = [int(f.name.split('_')[1]) for f in self.subfolders]
        self.time_step, self.time, self.subfolders = zip(
            *sorted(zip(self.time_step, self.time, self.subfolders)))

        #print(self.subfolders)

    def greet(self):
        print("I'm the simulation in folder " + self.folder)

    def get_data(self, string=None):
        if string is None:
            self.subfolders_slice = self.subfolders
        else:
            if ':' not in string:
                string = string + ':' + str(int(string)+1)
            exec("self.subfolders_slice = self.subfolders[" + string + "]")

        arrays = [np.loadtxt(os.path.join(f, "statevector.abpcsv"), skiprows=0)
                  for f in self.subfolders_slice]

        return np.vstack(arrays)

    def get_movement_data(self, string=None):
        if string is None:
            self.subfolders_slice = self.subfolders
        else:
            if ':' not in string:
                string = string + ':' + str(int(string)+1)
            exec("self.subfolders_slice = self.subfolders[" + string + "]")

        arrays = [np.loadtxt(os.path.join(f, "movement.abpcsv"), skiprows=0)
                  for f in self.subfolders_slice]

        return np.vstack(arrays)

class abpconf:
    def __init__(self, abpconffile, obstaclesfile, segmentsfile):
        if not os.path.isfile(abpconffile):
            raise Exception("Couldn't find abp.conf")

        self.conf_dict = {}
        exclude = [';', ' ', '\n']
        self.variations = []

        with open(abpconffile, 'r') as file:
            for line in file:
                if line[0] not in exclude:
                    key, value = line.split('=')
                    self.conf_dict[key.strip()] = value.strip()

        with open(obstaclesfile, 'r') as file:
            self.obstacles = file.read()

        with open(segmentsfile, 'r') as file:
            self.segments = file.read()


    def writeabp(self, file):
        with open(file, 'w') as file:
            for key, value in self.conf_dict.items():
                file.write(str(key) +  "=" +  str(value) + "\n")

    def writeobstacles(self, file):
        with open(file, 'w') as file:
            file.write(self.obstacles)

    def writesegments(self, file):
        with open(file, 'w') as file:
            file.write(self.segments)

    def add_numerical_variation(self, key, low, high, count):
        self.variations.append([key, np.linspace(low,high,count)])

    def print_all_variations(self):
        keyed = [[[item[0], value] for value in item[1]] for item in self.variations]
        jobs = list(itertools.product(*keyed))

        for j in jobs:
            print(j)

    def create_all_variations(self, folder):
        keyed = [[[item[0], value] for value in item[1]] for item in self.variations]
        jobs = list(itertools.product(*keyed))

        try:
            rmtree(folder)
        except:
            pass
        os.mkdir(folder)


        for i,j in enumerate(jobs):
            os.mkdir(folder + "/job" + str(i))
            for key, value in j:
                self.conf_dict[key.strip()] = value

            self.writeabp(folder + "/job" + str(i) + "/abp.conf")
            self.writeobstacles(folder + "/job" + str(i) + "/obstacles.conf")
            self.writesegments(folder + "/job" + str(i) + "/segments.conf")

def abp_run_all(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]

    for jobfolder in subfolders:
        print("Now running simulation in " + jobfolder)
        p = subprocess.Popen('abp', cwd=jobfolder)
        p.wait()
