import time
from pynvml import *

#monitor energy unit (J)
PACKAGE_MAX = 262143328850
DRAM_MAX = 65712999613

class energyModel:
    def __init__(self):
        self.static_power_GPU = 0 #32W (J)
        self.static_power_PACKAGE = 0
        self.static_power_DRAM = 0
        self.fd_PACKAGE = 0
        self.fd_DRAM = 0
        self.fd_GPU = 0
        self.start_powerDRAM = 0
        self.start_powerPACK = 0
        self.start_powerGPU = 0
        self.start_time = 0
        self.end_powerDRAM = 0
        self.end_powerPACK = 0
        self.end_powerGPU = 0
        self.end_time = 0

        print(f"STATIC PACKAGE E: {self.static_power_PACKAGE:.9f}")
        print(f"STATIC DRAM E: {self.static_power_DRAM:.9f}")
        print(f"STATIC GPU E: {self.static_power_GPU:.9f}")   

    #get profiled energy consumption of each operation
    def measure_virutal_ENERGY(self, option):
        #option == 0: optimization
        #option == 1: embedding layer
        #option == 2: preference extraction
        #option == 3: cache search
        #option == 4: cache insert
        #option == 5: hidden layer
        #option == 6: cache saving

        #return: latency, Package energy, DRAM energy, GPU energy

        if(option == 0):
            latency, package_E, dram_E, gpu_E = 3.93688624, 0.118618823612245, 0.000165424571428571, 0
        elif(option == 1):
            latency, package_E, dram_E, gpu_E = 0.000264072, 0.003745113, 1.88455E-05, 0.023468475
        elif(option == 2):
            latency, package_E, dram_E, gpu_E = 0.000977661, 0.012537900, 0.000004416, 0
        elif(option == 3):
            latency, package_E, dram_E, gpu_E = 0.000130583, 0.005441205, 7.25201E-06, 0
        elif(option == 4):
            latency, package_E, dram_E, gpu_E = 0.000307457, 0.003347516, 0.000013181, 0
        elif(option == 5):
            latency, package_E, dram_E, gpu_E = 0.003246384, 0.043375678, 0.00022245, 0.310933862
        else:
            latency, package_E, dram_E, gpu_E = 0, 0, 3.4308, 0
        
        return latency, package_E, dram_E, gpu_E
    

#real energy measure functions
#usage
#1. open RAPL file and get NVML driver by using "open_RAPL", "open_GPU"
#2. get static energy consumption of CPU, DRAM, GPU by using "init_STATIC_RAPL", "init_STATIC_GPU"
#3. use "general monitor" to measure total energy consumption, input appropriate "resource_op", "start_op"argument
#4. get dynamic energy consumption by using "measure_actual_ENERGY"
##################################################################################################################
    def drop_cache(self):
        try:
            os.system("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >> /dev/null")
            time.sleep(1)
        except Exception as e:
            print(f"Error dropping page cache: {e}")
    
    def open_RAPL(self, node: int):
        self.fd_DRAM = f'/sys/class/powercap/intel-rapl/intel-rapl:{node}/intel-rapl:{node}:0/energy_uj'
        self.fd_PACKAGE = f'/sys/class/powercap/intel-rapl/intel-rapl:{node}/energy_uj'

    def open_GPU(self, node: int):
        nvmlInit()
        self.fd_GPU = node

    def init_STATIC_RAPL(self):

        diff_package_list = []
        diff_dram_list = []

        for _ in range(2):
            self.monitor_TIME(0)
            self.monitor_RAPL(0)
            time.sleep(1)
            self.monitor_TIME(1)
            self.monitor_RAPL(1)
            diff_package_list.append(round((self.end_powerPACK - self.start_powerPACK) / (self.end_time - self.start_time),3))
            diff_dram_list.append(round((self.end_powerDRAM - self.start_powerDRAM) / (self.end_time - self.start_time),3))

        self.static_power_PACKAGE = sum(diff_package_list) / len(diff_package_list)
        self.static_power_DRAM = sum(diff_dram_list) / len(diff_dram_list)

        print(f"STATIC PACKAGE E: {self.static_power_PACKAGE:.9f}")
        print(f"STATIC DRAM E: {self.static_power_DRAM:.9f}")

    def init_STATIC_GPU(self):
        diff_list = []

        for _ in range(2):
            self.monitor_TIME(0)
            self.monitor_GPU(0)
            time.sleep(1)
            self.monitor_TIME(1)
            self.monitor_GPU(1)
            diff_list.append(round((self.end_powerGPU - self.start_powerGPU) / (self.end_time - self.start_time),3))
        self.static_power_GPU = sum(diff_list[1:]) / len(diff_list[1:])
        
        print(f"STATIC GPU E: {self.static_power_GPU:.9f}")


    def monitor_RAPL(self, option):
        fd_DRAM = open(self.fd_DRAM, 'r')
        fd_PACKAGE = open(self.fd_PACKAGE, 'r')

        if(not option):
            self.start_powerDRAM = (int(fd_DRAM.read())) / 1000000
            self.start_powerPACK = (int(fd_PACKAGE.read())) / 1000000
        else:
            self.end_powerDRAM = (int(fd_DRAM.read())) / 1000000
            self.end_powerPACK = (int(fd_PACKAGE.read())) / 1000000

        fd_DRAM.close()
        fd_PACKAGE.close()

    def monitor_GPU(self, option):
        nvmlDriver = nvmlDeviceGetHandleByIndex(self.fd_GPU)
        if(not option):
            self.start_powerGPU = (int(nvmlDeviceGetTotalEnergyConsumption(nvmlDriver))) / 1000
        else:
            self.end_powerGPU = (int(nvmlDeviceGetTotalEnergyConsumption(nvmlDriver))) / 1000

    def monitor_TIME(self, option):
        if(not option):
            self.start_time = round(time.time(), 6)
        else:
            self.end_time = round(time.time(), 6)
        
    def general_monitor(self, resouce_op, start_op): #start_op(0: start, 1: end)
        if(resouce_op == 0): #meausre only CPU, DRAM
            self.monitor_TIME(start_op)
            self.monitor_RAPL(start_op)
        else: #meausre only CPU, DRAM, GPU
            self.monitor_TIME(start_op)
            self.monitor_RAPL(start_op)
            self.monitor_GPU(start_op)
    
    def measure_actual_ENERGY(self, resouce_op):
        if(resouce_op == 0):
            if((self.end_powerPACK - self.start_powerPACK) < 0):
                print(f"overflow!!!: {(self.end_powerPACK - self.start_powerPACK)}")
            if((self.end_powerDRAM - self.start_powerDRAM) < 0):
                print(f"overflow!!!: {(self.end_powerDRAM - self.start_powerDRAM)}")

            latency = (self.end_time - self.start_time)
            package_E = (max(0, (self.end_powerPACK - self.start_powerPACK) - (self.static_power_PACKAGE * (self.end_time - self.start_time))))
            dram_E = (max(0, (self.end_powerDRAM - self.start_powerDRAM) - (self.static_power_DRAM * (self.end_time - self.start_time))))
            return latency, package_E, dram_E, 0
        else:
            if((self.end_powerPACK - self.start_powerPACK) < 0):
                print(f"overflow!!!: {(self.end_powerPACK - self.start_powerPACK)}")
            if((self.end_powerDRAM - self.start_powerDRAM) < 0):
                print(f"overflow!!!: {(self.end_powerDRAM - self.start_powerDRAM)}")

            latency = (self.end_time - self.start_time)
            package_E = (max(0, (self.end_powerPACK - self.start_powerPACK) - (self.static_power_PACKAGE * (self.end_time - self.start_time))))
            dram_E = (max(0, (self.end_powerDRAM - self.start_powerDRAM) - (self.static_power_DRAM * (self.end_time - self.start_time))))
            gpu_E = (max(0, (self.end_powerGPU - self.start_powerGPU) - (self.static_power_GPU * (self.end_time - self.start_time))))
            return latency, package_E, dram_E, gpu_E
##################################################################################################################